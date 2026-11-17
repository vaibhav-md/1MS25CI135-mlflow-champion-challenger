import os
import json
import time
import torch
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from importlib.metadata import version
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

os.makedirs("outputs", exist_ok=True)

# Configuration
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "Sentiment_Champion_Challenger"
STUDENT_USN = "1MS25CI135"
SEED = 42
MARGIN = 0.01

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"

df = pd.read_csv("data/eval_inputs.csv")
texts = df["text"].astype(str).tolist()
y_true = df["true_label"].tolist()

models = {
    "model_A": "distilbert-base-uncased-finetuned-sst-2-english",
    "model_B": "textattack/bert-base-uncased-SST-2"
}

results = {}
predictions = {}

# Evaluate each model
for name, model_id in models.items():
    role = "champion_candidate" if name == "model_A" else "challenger"

    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        device=device
    )

    classifier(texts[:5])  # Warm-up
    start = time.perf_counter()
    output = classifier(texts, batch_size=16)
    elapsed = time.perf_counter() - start

    labels = [
        ("POSITIVE" if x["label"] == "LABEL_1" else "NEGATIVE")
        if name == "model_B" else x["label"]
        for x in output
    ]

    predictions[name] = labels

    metrics = {
        "accuracy": accuracy_score(y_true, labels),
        "precision": precision_score(
            y_true, labels, pos_label="POSITIVE", zero_division=0
        ),
        "recall": recall_score(
            y_true, labels, pos_label="POSITIVE", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, labels, average="macro", zero_division=0
        ),
        "total_inference_seconds": elapsed,
        "avg_latency_ms": elapsed / len(texts) * 1000
    }

    results[name] = metrics

    with mlflow.start_run(
        run_name=f"{'A' if name == 'model_A' else 'B'}_sentiment_{name}"
    ):
        mlflow.log_params({
            "hf_model_id": model_id,
            "task": "sentiment-analysis",
            "role": role,
            "evaluation_samples": len(texts),
            "seed": SEED,
            "dataset_name": "GLUE SST-2 validation",
            "device": device_name,
            "batch_size": 16,
            "transformers_version": version("transformers"),
            "torch_version": torch.__version__,
            "sklearn_version": version("scikit-learn"),
            "pandas_version": pd.__version__
        })

        mlflow.log_metrics(metrics)

        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "dataset_name": "GLUE SST-2 validation",
            "model_role": role
        })

        mlflow.log_artifact("data/eval_inputs.csv")

# Save comparison predictions
comparison = df.copy()
comparison["model_A_prediction"] = predictions["model_A"]
comparison["model_B_prediction"] = predictions["model_B"]

comparison.to_csv(
    "outputs/model_comparison_predictions.csv",
    index=False
)

# Save five qualitative disagreement examples
disagreements = comparison[
    comparison["model_A_prediction"] != comparison["model_B_prediction"]
].head(5).copy()

def explain(row):
    a_correct = row["model_A_prediction"] == row["true_label"]
    b_correct = row["model_B_prediction"] == row["true_label"]

    if a_correct and not b_correct:
        return "Model A matches the reference label."
    if b_correct and not a_correct:
        return "Model B matches the reference label."
    return "Both models agree on correctness relative to the reference."

disagreements["analysis"] = disagreements.apply(explain, axis=1)
disagreements.to_csv(
    "outputs/disagreement_examples.csv",
    index=False
)

# Decide Champion using the predefined rule
delta = results["model_B"]["macro_f1"] - results["model_A"]["macro_f1"]

if delta >= MARGIN:
    winner = "model_B"
    reason = "Challenger improved macro-F1 by at least 0.01."
else:
    winner = min(
        results,
        key=lambda name: results[name]["total_inference_seconds"]
    )
    reason = (
        "Challenger did not meet the 0.01 macro-F1 improvement threshold; "
        "the faster model was selected."
    )

summary = {
    "model_A": results["model_A"],
    "model_B": results["model_B"],
    "decision": {
        "primary_metric": "macro_f1",
        "minimum_improvement_margin": MARGIN,
        "macro_f1_delta_B_minus_A": delta,
        "winner": winner,
        "reason": reason,
        "device": device_name,
        "evaluation_samples": len(texts)
    }
}

with open("outputs/metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

# Metric comparison plot
pd.DataFrame(results).loc[
    ["accuracy", "precision", "recall", "macro_f1"]
].plot(kind="bar", figsize=(9, 5))

plt.title("Model Metric Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/metric_comparison.png", dpi=300)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, name in zip(axes, ["model_A", "model_B"]):
    cm = confusion_matrix(
        y_true,
        predictions[name],
        labels=["NEGATIVE", "POSITIVE"]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["NEGATIVE", "POSITIVE"],
        yticklabels=["NEGATIVE", "POSITIVE"]
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=300)
plt.close()

# Log comparison summary and all output artifacts
with mlflow.start_run(run_name="compare_sentiment"):
    mlflow.log_params({
        "evaluation_samples": len(texts),
        "primary_metric": "macro_f1",
        "minimum_improvement_margin": MARGIN,
        "selected_champion": winner,
        "device": device_name,
        "seed": SEED
    })

    mlflow.set_tags({
        "student_usn": STUDENT_USN,
        "dataset_name": "GLUE SST-2 validation",
        "run_role": "comparison_summary"
    })

    summary_metrics = {}
    for model_name, model_metrics in results.items():
        for metric_name, value in model_metrics.items():
            summary_metrics[f"{model_name}_{metric_name}"] = value

    mlflow.log_metrics(summary_metrics)
    mlflow.log_metric("macro_f1_delta_B_minus_A", delta)
    mlflow.log_artifacts("outputs")

print(json.dumps(summary, indent=4))
print("Evaluation completed. Results and artifacts saved.")
