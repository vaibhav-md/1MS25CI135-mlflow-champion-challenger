import json
import pandas as pd
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
REGISTERED_NAME = "SentimentModel"

class SentimentWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_id, map_labels=False):
        self.model_id = model_id
        self.map_labels = map_labels

    def load_context(self, context):
        from transformers import pipeline
        self.classifier = pipeline("sentiment-analysis", model=self.model_id)

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        output = self.classifier(texts)
        labels = [
            ("POSITIVE" if x["label"] == "LABEL_1" else "NEGATIVE")
            if self.map_labels else x["label"]
            for x in output
        ]
        return pd.DataFrame({
            "prediction": labels,
            "confidence": [x["score"] for x in output]
        })

models = [
    ("Model_A", "distilbert-base-uncased-finetuned-sst-2-english", False),
    ("Model_B", "textattack/bert-base-uncased-SST-2", True)
]

for name, model_id, map_labels in models:
    with mlflow.start_run(run_name=f"Register_{name}"):
        info = mlflow.pyfunc.log_model(
            name=name.lower(),
            python_model=SentimentWrapper(model_id, map_labels),
            registered_model_name=REGISTERED_NAME,
            input_example=pd.DataFrame({"text": ["This is a good experience."]})
        )

    versions = client.search_model_versions(
        f"name='{REGISTERED_NAME}'"
    )
    version = max(versions, key=lambda v: int(v.version))
    client.set_model_version_tag(
        REGISTERED_NAME, version.version, "hf_model_id", model_id
    )
    print(f"{name} registered as version {version.version}")

with open("metrics_summary.json") as f:
    metrics = json.load(f)

a = metrics["Model_A"]
b = metrics["Model_B"]

# Predefined rule: Challenger must improve macro-F1 by at least 0.01.
# Otherwise, the faster model wins.
if b["macro_f1"] >= a["macro_f1"] + 0.01:
    champion_version = "2"
    challenger_version = "1"
else:
    champion_version = "1" if a["inference_time"] <= b["inference_time"] else "2"
    challenger_version = "2" if champion_version == "1" else "1"

client.set_registered_model_alias(
    REGISTERED_NAME, "champion", champion_version
)
client.set_registered_model_alias(
    REGISTERED_NAME, "challenger", challenger_version
)

print("Champion version:", champion_version)
print("Challenger version:", challenger_version)
print("Aliases assigned successfully.")
