import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/reddit_with_mention_regime.csv")
df["text"] = df["text"].astype(str)

df1 = df[df["mention_regime"] == 1].sample(n=150, random_state=42)
df0 = df[df["mention_regime"] == 0].sample(n=150, random_state=43)

sample = pd.concat([df1, df0]).sample(frac=1, random_state=42)

output_path = Path("data/labeling/label_sample.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)
sample.to_json(output_path, orient="records", lines=True, force_ascii=False)
print(f"Saved {len(sample)} records to {output_path}")
