import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

REGISTERED_NAME = "SentimentModel"

models = [
    ("model_A", "distilbert-base-uncased-finetuned-sst-2-english", False),
    ("model_B", "textattack/bert-base-uncased-SST-2", True)
]

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

try:
    client.get_registered_model(REGISTERED_NAME)
except MlflowException:
    client.create_registered_model(REGISTERED_NAME)

versions = {}

for name, model_id, map_labels in models:
    existing = [
        v for v in client.search_model_versions(f"name='{REGISTERED_NAME}'")
        if v.tags.get("hf_model_id") == model_id
    ]

    if existing:
        version = max(existing, key=lambda v: int(v.version))
        print(f"{name}: reusing version {version.version}")
    else:
        before = {
            v.version for v in
            client.search_model_versions(f"name='{REGISTERED_NAME}'")
        }

        with mlflow.start_run(run_name=f"Register_{name}"):
            mlflow.pyfunc.log_model(
                name=name,
                python_model=SentimentWrapper(model_id, map_labels),
                registered_model_name=REGISTERED_NAME,
                input_example=pd.DataFrame({
                    "text": ["This is a good experience."]
                })
            )

        after = client.search_model_versions(
            f"name='{REGISTERED_NAME}'"
        )

        created = [v for v in after if v.version not in before]

        if not created:
            raise RuntimeError(f"Could not find newly registered version for {name}")

        version = max(created, key=lambda v: int(v.version))

        client.set_model_version_tag(
            REGISTERED_NAME, version.version, "hf_model_id", model_id
        )

        print(f"{name}: registered version {version.version}")

    versions[model_id] = version.version

model_a = "distilbert-base-uncased-finetuned-sst-2-english"
model_b = "textattack/bert-base-uncased-SST-2"

client.set_registered_model_alias(
    REGISTERED_NAME, "champion", versions[model_a]
)
client.set_registered_model_alias(
    REGISTERED_NAME, "challenger", versions[model_b]
)

print("Champion → Model A, version", versions[model_a])
print("Challenger → Model B, version", versions[model_b])
