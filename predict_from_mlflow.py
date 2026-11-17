import os
os.makedirs('outputs', exist_ok=True)

import argparse
import pandas as pd
import mlflow
from mlflow import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/new_inputs.csv")
parser.add_argument("--output", default="outputs/new_data_predictions.csv")
args = parser.parse_args()

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_name = "SentimentModel"
client = MlflowClient()

version = client.get_model_version_by_alias(model_name, "champion")
model_id = version.tags.get("hf_model_id", "Not recorded")

model_uri = f"models:/{model_name}@champion"
model = mlflow.pyfunc.load_model(model_uri)

print("Champion version:", version.version)
print("Hugging Face model:", model_id)

data = pd.read_csv(args.input)
predictions = model.predict(data[["text"]])

output = data.copy()
output["prediction"] = predictions["prediction"].values
output["confidence"] = predictions["confidence"].values

output.to_csv(args.output, index=False)

print("\\nPredictions:")
print(output.to_string(index=False))
print("\\nSaved:", args.output)
