# Healthcare Information Storage and Retrieval

## Run
1. Open this folder in VS Code.
2. Ensure Java JDK and Python are installed.
3. In the VS Code terminal run:
   `python main.py`

## Requirements
`pip install pyspark`

## Output files
The program creates an `outputs` folder containing:
- `report.txt` — complete readable output
- `positional_index.json`
- `raw_tf.csv`
- `weighted_tf.csv`
- `tfidf.csv`
- `normalized_tfidf.csv`

The positional index is built using a Spark RDD. Phrase matching uses exact contiguous positions.
