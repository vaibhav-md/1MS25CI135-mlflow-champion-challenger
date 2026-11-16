# Project Notes

## Task
Sentiment Analysis using a Champion–Challenger MLOps workflow.

## Models Compared
- Champion candidate: distilbert-base-uncased-finetuned-sst-2-english
- Challenger candidate: textattack/bert-base-uncased-SST-2

## Evaluation
Both models were evaluated on the same labelled dataset.

- Accuracy: 1.00 for both
- Macro F1: 1.00 for both
- Model A inference time: approximately 0.93 seconds
- Model B inference time: approximately 2.48 seconds

## Selection
Model A was selected as Champion because both models achieved equal evaluation metrics, while Model A was faster.

## Note
The evaluation dataset contains only 10 examples. These results are preliminary and should not be treated as evidence of general model performance.
