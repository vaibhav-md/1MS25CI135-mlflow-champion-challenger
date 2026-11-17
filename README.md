# Champion–Challenger Model Comparison

## Project Overview
This project compares two Hugging Face sentiment-classification models using identical evaluation inputs, MLflow tracking, and model registry aliases.

## Models
- Model A: `distilbert-base-uncased-finetuned-sst-2-english`
- Model B: `textattack/bert-base-uncased-SST-2`

## Dataset and Evaluation
Both models were evaluated on 200 labelled sentences from the GLUE SST-2 validation dataset.

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | 0.9100 | 0.9150 |
| Precision | 0.8716 | 0.8942 |
| Recall | 0.9596 | 0.9394 |
| Macro-F1 | 0.9099 | 0.9150 |
| Inference time (seconds) | 13.7653 | 27.4046 |

## Champion Selection and Decision Rationale
Model B achieved a macro-F1 of 0.9150, compared with 0.9099 for Model A. Its improvement of approximately 0.0051 is below the predefined minimum improvement margin of 0.01. Therefore, Model B does not qualify to replace the Champion.

Model A completed inference in approximately 13.77 seconds, whereas Model B required approximately 27.40 seconds on the same 200 evaluation samples. Model A was therefore about twice as fast. Model B achieved slightly higher precision, while Model A achieved higher recall. Since the Challenger did not meet the required macro-F1 improvement threshold, the faster model was retained. Model A remains the Champion and Model B remains the Challenger. This decision follows the predefined selection rule rather than relying on a single metric. These results are limited to the selected 200-sample evaluation subset and do not guarantee performance on every dataset.

## MLflow Registry
- Tracking URI: `sqlite:///mlflow.db`
- Experiment: `Sentiment_Champion_Challenger`
- Registered model: `SentimentModel`
- Champion alias: Version 1 (Model A)
- Challenger alias: Version 2 (Model B)

The Champion was loaded using its MLflow alias and used to predict 10 unseen examples. Results are saved in `outputs/new_data_predictions.csv`.

## Project Files
- `evaluate.py`: evaluates both models and logs results.
- `register_champion.py`: registers models and assigns aliases.
- `predict_from_mlflow.py`: loads the Champion alias for predictions.
- `data/`: evaluation and unseen input CSV files.
- `outputs/`: metrics, predictions, disagreement examples and plots.
- `screenshots/`: MLflow comparison, registry aliases and prediction output.

## Running the Project
Run from the repository root:

    pip install -r requirements.txt
    python evaluate.py
    python register_champion.py
    python predict_from_mlflow.py

The MLflow database and local model artifacts are intentionally excluded from GitHub.

## Tools
Python, Hugging Face Transformers, PyTorch, scikit-learn, pandas, Matplotlib, Seaborn and MLflow.

AI assistance was used during development and debugging. The implementation and evaluation decisions should be understood and explainable by the student.