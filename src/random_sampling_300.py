import pandas as pd

df = pd.read_csv("data/processed/reddit_with_mention_regime.csv")
df["text"] = df["text"].astype(str)

df1 = df[df["mention_regime"] == 1].sample(n=150)
df0 = df[df["mention_regime"] == 0].sample(n=150)

sample = pd.concat([df1, df0])

sample.to_json("data/labeling/label_sample.json", orient="records", lines=True, force_ascii=False)
#load the csv to json, transform into lines, and keep non ascii strings
