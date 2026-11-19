# Project Notes

## Task
Sentiment Analysis using a Champion–Challenger MLOps workflow.

## Models Compared
- Model A: `distilbert-base-uncased-finetuned-sst-2-english`
- Model B: `textattack/bert-base-uncased-SST-2`

## Evaluation
Both models were evaluated on the same 200 labelled sentences from the GLUE SST-2 validation dataset.

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | 0.9100 | 0.9150 |
| Precision | 0.8716 | 0.8942 |
| Recall | 0.9596 | 0.9394 |
| Macro-F1 | 0.9099 | 0.9150 |
| Inference Time (s) | 13.7653 | 27.4046 |

## Champion Selection
Model B achieved a slightly higher Macro-F1 score, but its improvement of 0.0051 was below the predefined improvement margin of 0.01.

Model A was approximately twice as fast during inference. Therefore, Model A was retained as Champion, while Model B remained the Challenger.

## MLflow Registry
- Experiment: `Sentiment_Champion_Challenger`
- Registered model: `SentimentModel`
- Champion alias: Version 1 (Model A)
- Challenger alias: Version 2 (Model B)

## Prediction
The Champion model was loaded through its MLflow alias and used to predict sentiment for 10 unseen examples. Predictions were saved in `outputs/new_data_predictions.csv`.

## Limitation
The evaluation was performed on a selected 200-sample subset. Results may differ on other datasets or real-world inputs. Individual misclassifications are possible.
