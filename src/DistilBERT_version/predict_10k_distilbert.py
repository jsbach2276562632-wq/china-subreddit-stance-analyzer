import pandas as pd
import torch
from pathlib import Path
import os
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification


df = pd.read_csv("data/processed/reddit_with_mention_regime.csv")
df["text"] = df["text"].astype(str)

model_path = "models/distilbert"

print("model path:", model_path)
print("exists:", Path(model_path).exists())
print("files:", os.listdir(model_path))

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)



label_map = {
    0: "critical",
    1: "neutral",
    2: "positive"
}

model.eval() # evaluation mode

predictions = []


for text in df["text"]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    predictions.append(label_map[pred_id])


df["predicted_label"] = predictions

df.to_csv("data/processed/reddit_with_mention_regime_labeled_distilbert.csv", index=False, encoding="utf-8-sig")

print("saved")
