import json
import time
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Sentiment_Champion_Challenger")

df = pd.read_csv("eval_inputs.csv")
texts = df["text"].astype(str).tolist()
y_true = df["true_label"].tolist()

models = {
    "Model_A": "distilbert-base-uncased-finetuned-sst-2-english",
    "Model_B": "textattack/bert-base-uncased-SST-2"
}

results = {}
predictions = {}

for name, model_id in models.items():
    classifier = pipeline("sentiment-analysis", model=model_id)

    start = time.time()
    output = classifier(texts)
    elapsed = time.time() - start

    labels = [
        ("POSITIVE" if x["label"] == "LABEL_1" else "NEGATIVE")
        if name == "Model_B" else x["label"]
        for x in output
    ]

    predictions[name] = labels

    metrics = {
        "accuracy": accuracy_score(y_true, labels),
        "precision": precision_score(y_true, labels, average="macro", zero_division=0),
        "recall": recall_score(y_true, labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, labels, average="macro", zero_division=0),
        "inference_time": elapsed
    }
    results[name] = metrics

    with mlflow.start_run(run_name=name):
        mlflow.log_param("hf_model_id", model_id)
        mlflow.log_param("evaluation_samples", len(texts))
        mlflow.log_metrics(metrics)
        mlflow.set_tag("task", "sentiment-analysis")
        mlflow.log_artifact("eval_inputs.csv")

comparison = df.copy()
comparison["model_A_prediction"] = predictions["Model_A"]
comparison["model_B_prediction"] = predictions["Model_B"]
comparison.to_csv("model_comparison_predictions.csv", index=False)

with open("metrics_summary.json", "w") as f:
    json.dump(results, f, indent=4)

# Metric comparison plot
metric_names = ["accuracy", "precision", "recall", "macro_f1"]
plot_df = pd.DataFrame(results).loc[metric_names]
plot_df.plot(kind="bar", figsize=(9, 5))
plt.title("Model Metric Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("metric_comparison.png")
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, ["Model_A", "Model_B"]):
    cm = confusion_matrix(y_true, predictions[name], labels=["NEGATIVE", "POSITIVE"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["NEGATIVE", "POSITIVE"],
                yticklabels=["NEGATIVE", "POSITIVE"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.close()

print(json.dumps(results, indent=4))
print("Evaluation completed.")
