from pathlib import Path
from collections import Counter, defaultdict
from math import log10, sqrt
import json
import csv
import sys

from pyspark.sql import SparkSession


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
N_EXPECTED = 10

PHRASE_1 = "patient care"
PHRASE_2 = "medical research"


def find_documents():
    """Read d1.txt ... d10.txt from project root or ./data."""
    candidates = [BASE_DIR, BASE_DIR / "data"]
    found = {}
    for folder in candidates:
        if all((folder / f"d{i}.txt").is_file() for i in range(1, 11)):
            for i in range(1, 11):
                found[f"d{i}"] = (folder / f"d{i}.txt").read_text(
                    encoding="utf-8"
                ).strip()
            return found
    missing = [
        f"d{i}.txt"
        for i in range(1, 11)
        if not any((folder / f"d{i}.txt").is_file() for folder in candidates)
    ]
    raise FileNotFoundError(
        "Could not find all ten documents. Put d1.txt ... d10.txt "
        f"in {BASE_DIR} or in {BASE_DIR / 'data'}. Missing: {missing}"
    )


def tokenize(text):
    # Dataset instructions: lowercase and split by whitespace.
    return text.lower().split()


def build_positional_index(spark, documents):
    """
    Build the positional index using a Spark RDD.
    Each token position is 1-based within its document.
    """
    rows = list(documents.items())
    docs_rdd = spark.sparkContext.parallelize(rows, min(len(rows), 4))

    term_positions = (
        docs_rdd
        .flatMap(
            lambda item: [
                (term, (item[0], position))
                for position, term in enumerate(tokenize(item[1]), start=1)
            ]
        )
        .groupByKey()
        .mapValues(list)
        .collect()
    )

    index = defaultdict(lambda: defaultdict(list))
    for term, occurrences in term_positions:
        for doc_id, position in occurrences:
            index[term][doc_id].append(position)

    # Stable alphabetical/document ordering.
    clean_index = {}
    for term in sorted(index):
        clean_index[term] = {}
        for doc_id in sorted(index[term], key=lambda x: int(x[1:])):
            clean_index[term][doc_id] = sorted(index[term][doc_id])
    return clean_index


def build_statistics(documents, positional_index):
    doc_ids = sorted(documents, key=lambda x: int(x[1:]))
    terms = sorted(positional_index)

    raw_tf = {term: {} for term in terms}
    df = {}
    idf = {}
    weighted_tf = {term: {} for term in terms}
    tfidf = {term: {} for term in terms}

    for term in terms:
        postings = positional_index[term]
        df[term] = len(postings)
        idf[term] = log10(N_EXPECTED / df[term])
        for doc_id in doc_ids:
            count = len(postings.get(doc_id, []))
            raw_tf[term][doc_id] = count
            wt = (1 + log10(count)) if count > 0 else 0.0
            weighted_tf[term][doc_id] = wt
            tfidf[term][doc_id] = wt * idf[term]

    doc_lengths = {}
    normalized = {term: {} for term in terms}
    for doc_id in doc_ids:
        length = sqrt(sum(tfidf[t][doc_id] ** 2 for t in terms))
        doc_lengths[doc_id] = length
        for term in terms:
            normalized[term][doc_id] = (
                tfidf[term][doc_id] / length if length else 0.0
            )

    return {
        "doc_ids": doc_ids,
        "terms": terms,
        "raw_tf": raw_tf,
        "df": df,
        "idf": idf,
        "weighted_tf": weighted_tf,
        "tfidf": tfidf,
        "doc_lengths": doc_lengths,
        "normalized": normalized,
    }


def phrase_matches(phrase, positional_index):
    """Exact contiguous phrase matching using 1-based token positions."""
    phrase_terms = tokenize(phrase)
    if not phrase_terms:
        return {}

    first_term = phrase_terms[0]
    candidates = positional_index.get(first_term, {})
    matches = {}

    for doc_id, first_positions in candidates.items():
        starts = []
        for start in first_positions:
            is_match = True
            for offset, term in enumerate(phrase_terms[1:], start=1):
                term_positions = positional_index.get(term, {}).get(doc_id, [])
                if (start + offset) not in term_positions:
                    is_match = False
                    break
            if is_match:
                starts.append(start)
        if starts:
            matches[doc_id] = starts
    return matches


def query_similarity(query, stats):
    """Cosine similarity between normalized query TF-IDF and doc vectors."""
    q_terms = tokenize(query)
    q_tf = Counter(q_terms)
    q_weights = {}
    for term, count in q_tf.items():
        if term not in stats["idf"]:
            continue
        weighted = 1 + log10(count) if count > 0 else 0.0
        q_weights[term] = weighted * stats["idf"][term]

    q_length = sqrt(sum(value * value for value in q_weights.values()))
    if q_length:
        q_norm = {term: value / q_length for term, value in q_weights.items()}
    else:
        q_norm = {}

    scores = {}
    for doc_id in stats["doc_ids"]:
        score = 0.0
        for term, q_value in q_norm.items():
            score += q_value * stats["normalized"].get(term, {}).get(doc_id, 0.0)
        scores[doc_id] = score
    return q_length, q_weights, q_norm, scores


def ranked_scores(doc_ids, scores):
    return sorted(doc_ids, key=lambda d: (-scores.get(d, 0.0), int(d[1:])))


def fmt(value):
    return f"{value:.4f}"


def matrix_text(title, terms, doc_ids, matrix):
    lines = [f"\n{title}", "Term".ljust(16) + "".join(d.ljust(9) for d in doc_ids)]
    for term in terms:
        row = term.ljust(16) + "".join(fmt(matrix[term][d]).ljust(9) for d in doc_ids)
        lines.append(row)
    return "\n".join(lines)


def make_report(documents, index, stats):
    out = []
    doc_ids = stats["doc_ids"]
    terms = stats["terms"]

    out.append("=" * 90)
    out.append("HEALTHCARE INFORMATION STORAGE AND RETRIEVAL PROJECT")
    out.append("Spark Positional Index, Retrieval and Ranking")
    out.append("=" * 90)
    out.append("\nSPARK EXECUTION")
    out.append("Spark application: local[*]")
    out.append(f"Documents processed: {len(documents)}")
    out.append(f"Unique terms: {len(terms)}")

    out.append("\n\nASSIGNED DATASET")
    for doc_id in doc_ids:
        out.append(f"{doc_id}: {documents[doc_id]}")

    out.append("\n\nPART A — POSITIONAL INDEX (1-BASED POSITIONS)")
    for term in terms:
        postings = "; ".join(
            f"{doc_id}: {', '.join(map(str, index[term][doc_id]))}"
            for doc_id in index[term]
        )
        out.append(f"{term:<16} {postings}")

    out.append(matrix_text("RAW TERM FREQUENCY (TF) MATRIX", terms, doc_ids, stats["raw_tf"]))
    out.append(matrix_text("WEIGHTED TF MATRIX: 1 + log10(TF)", terms, doc_ids, stats["weighted_tf"]))

    out.append("\n\nDOCUMENT FREQUENCY (DF) AND INVERSE DOCUMENT FREQUENCY (IDF)")
    out.append(f"{'Term':<16} {'DF':<8} {'IDF':<10}")
    for term in terms:
        out.append(f"{term:<16} {stats['df'][term]:<8} {fmt(stats['idf'][term]):<10}")

    out.append(matrix_text("TF-IDF MATRIX", terms, doc_ids, stats["tfidf"]))

    out.append("\n\nDOCUMENT VECTOR LENGTHS (L2)")
    for doc_id in doc_ids:
        out.append(f"{doc_id}: {stats['doc_lengths'][doc_id]:.6f}")

    out.append(matrix_text("NORMALIZED TF-IDF MATRIX", terms, doc_ids, stats["normalized"]))

    # Phrase query reports
    phrase1_matches = phrase_matches(PHRASE_1, index)
    phrase2_matches = phrase_matches(PHRASE_2, index)

    out.append("\n\nPART B — PHRASE QUERIES")
    for phrase, matches in [(PHRASE_1, phrase1_matches), (PHRASE_2, phrase2_matches)]:
        out.append(f"\nPhrase query: {phrase!r}")
        if not matches:
            out.append("No matching documents.")
        else:
            for doc_id in sorted(matches, key=lambda x: int(x[1:])):
                ranges = ", ".join(
                    f"{start}-{start + len(tokenize(phrase)) - 1}"
                    for start in matches[doc_id]
                )
                out.append(
                    f"{doc_id}: starting positions {matches[doc_id]}; token ranges {ranges}"
                )

        q_len, q_weights, q_norm, scores = query_similarity(phrase, stats)
        out.append(f"Query vector length: {q_len:.6f}")
        out.append("Query term weights:")
        for term in tokenize(phrase):
            if term in q_weights:
                out.append(
                    f"  {term}: query TF-IDF={fmt(q_weights[term])}, "
                    f"normalized={fmt(q_norm[term])}"
                )
        if matches:
            order = ranked_scores(matches.keys(), scores)
            out.append("Similarity scores (matching documents):")
            for doc_id in sorted(matches, key=lambda x: int(x[1:])):
                out.append(f"  {doc_id}: {fmt(scores[doc_id])}")
            out.append("Ranking (highest first): " + " > ".join(order))

    # Boolean AND
    and_docs = set(phrase1_matches) & set(phrase2_matches)
    and_query = f"{PHRASE_1} {PHRASE_2}"
    q_len, q_weights, q_norm, and_scores = query_similarity(and_query, stats)
    and_order = ranked_scores(and_docs, and_scores)

    out.append(f"\n\nBOOLEAN AND: {PHRASE_1} AND {PHRASE_2}")
    out.append("Matching documents: " + (", ".join(sorted(and_docs, key=lambda x: int(x[1:]))) if and_docs else "None"))
    out.append(f"Combined query vector length: {q_len:.6f}")
    for doc_id in sorted(and_docs, key=lambda x: int(x[1:])):
        out.append(f"{doc_id}: similarity={fmt(and_scores[doc_id])}")
    out.append("Ranking: " + (" > ".join(and_order) if and_order else "None"))

    # Boolean AND NOT: score against the positive phrase query, as the excluded
    # phrase is a filter, not a positive relevance term.
    and_not_docs = set(phrase1_matches) - set(phrase2_matches)
    _, _, _, and_not_scores = query_similarity(PHRASE_1, stats)
    and_not_order = ranked_scores(and_not_docs, and_not_scores)

    out.append(f"\n\nBOOLEAN AND NOT: {PHRASE_1} AND NOT {PHRASE_2}")
    out.append("Matching documents: " + (", ".join(sorted(and_not_docs, key=lambda x: int(x[1:]))) if and_not_docs else "None"))
    for doc_id in sorted(and_not_docs, key=lambda x: int(x[1:])):
        out.append(f"{doc_id}: similarity={fmt(and_not_scores[doc_id])}")
    out.append("Ranking: " + (" > ".join(and_not_order) if and_not_order else "None"))

    out.append("\n\nCALCULATION CONVENTIONS")
    out.append("N = 10 documents.")
    out.append("Raw TF = number of occurrences of a term in a document.")
    out.append("DF = number of documents containing the term.")
    out.append("IDF = log10(N / DF).")
    out.append("Weighted TF = 1 + log10(Raw TF) when Raw TF > 0; otherwise 0.")
    out.append("TF-IDF = Weighted TF × IDF.")
    out.append("Document vector length = sqrt(sum of squared TF-IDF weights).")
    out.append("Normalized TF-IDF = TF-IDF / document vector length.")
    out.append("Query vectors use the same weighted TF and collection IDF, then L2 normalization.")
    out.append("Similarity = dot product of normalized query and normalized document vectors.")
    out.append("Phrase matching is exact and contiguous using the positional index.")
    out.append("AND = intersection of phrase match sets.")
    out.append("AND NOT = first phrase match set minus second phrase match set.")

    out.append("\\n\\nCONCEPTUAL EXPLANATIONS FOR THE REPORT")
    out.append("1. Positional information stores each term's exact token locations. A phrase matches only when its terms occur in consecutive positions in the same document.")
    out.append("2. DF counts how many documents contain a term. IDF = log10(N/DF), so a term found in many documents receives a lower IDF, while a rarer term receives a higher IDF.")
    out.append("3. Repeated occurrences increase raw TF. Weighted TF uses 1 + log10(TF), which increases the weight while reducing the effect of very frequent repetition.")
    out.append("4. TF-IDF combines within-document term frequency with collection-wide rarity, giving more importance to terms that characterize a document.")
    out.append("5. L2 normalization divides each document vector by its length. This reduces the influence of document length and makes cosine similarity comparable.")
    out.append("6. Similarity is the dot product of normalized query and document vectors. Higher scores indicate a closer match; results are ranked from highest to lowest.")
    out.append("7. AND returns documents that contain both exact phrases. AND NOT returns documents containing the first phrase while excluding documents containing the second phrase.")
    return "\\n".join(out), phrase1_matches, phrase2_matches


def save_csv(path, terms, doc_ids, matrix):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Term"] + doc_ids)
        for term in terms:
            writer.writerow([term] + [matrix[term][doc_id] for doc_id in doc_ids])


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    documents = find_documents()

    print("Starting Healthcare Spark project...")
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("HealthcareInformationRetrieval")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        positional_index = build_positional_index(spark, documents)
        stats = build_statistics(documents, positional_index)
        report, phrase1_matches, phrase2_matches = make_report(
            documents, positional_index, stats
        )

        # Save all required deliverables.
        (OUTPUT_DIR / "report.txt").write_text(report, encoding="utf-8")
        (OUTPUT_DIR / "positional_index.json").write_text(
            json.dumps(positional_index, indent=2), encoding="utf-8"
        )
        save_csv(OUTPUT_DIR / "raw_tf.csv", stats["terms"], stats["doc_ids"], stats["raw_tf"])
        save_csv(OUTPUT_DIR / "weighted_tf.csv", stats["terms"], stats["doc_ids"], stats["weighted_tf"])
        save_csv(OUTPUT_DIR / "tfidf.csv", stats["terms"], stats["doc_ids"], stats["tfidf"])
        save_csv(OUTPUT_DIR / "normalized_tfidf.csv", stats["terms"], stats["doc_ids"], stats["normalized"])

        print("\n" + report)
        print("\n\n[SUCCESS] All outputs saved in:", OUTPUT_DIR)
        print("[SUCCESS] Spark session completed.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
