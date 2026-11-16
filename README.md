# Champion–Challenger Model Comparison

## Project
This project compares two Hugging Face sentiment-classification models
using MLflow experiment tracking and model registry aliases.

## Models
- Model A: distilbert-base-uncased-finetuned-sst-2-english
- Model B: textattack/bert-base-uncased-SST-2

Both models are evaluated on the same labelled inputs.

## Evaluation
The primary metric is macro-F1. The predefined decision rule is:
1. The Challenger replaces the Champion if its macro-F1 improves by at least 0.01.
2. Otherwise, the faster model is selected.

The current evaluation uses 10 manually labelled examples. Both models
achieved perfect scores on this small set, so these results should not
be interpreted as evidence of general performance.

## MLflow
Experiment tracking URI: sqlite:///mlflow.db
Experiment: Sentiment_Champion_Challenger
Registered model: SentimentModel
Aliases: champion and challenger

## Files
- evaluate.py: evaluates both models and generates metrics and plots.
- register_champion.py: registers models and assigns aliases.
- predict_from_mlflow.py: loads the Champion through its MLflow alias.
- eval_inputs.csv: labelled evaluation data.
- new_inputs.csv: unseen prediction inputs.
- model_comparison_predictions.csv: evaluation predictions.
- new_data_predictions.csv: Champion predictions.
- metrics_summary.json: evaluation metrics.
- metric_comparison.png and confusion_matrices.png: evaluation plots.

## Run
Install dependencies:
```bash
pip install -r requirements.txt
```

Run evaluation:
```bash
python evaluate.py
```

Register models and aliases:
```bash
python register_champion.py
```

Predict using the Champion:
```bash
python predict_from_mlflow.py
```

The MLflow database and local model artifacts are intentionally excluded
from GitHub. Run the scripts to recreate them.
