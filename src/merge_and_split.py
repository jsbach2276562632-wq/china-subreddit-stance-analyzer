import json
import pandas as pd
from sklearn.model_selection import train_test_split


df_raw= pd.read_json("data/labeling/label_sample.jsonl", lines=True)
df_raw["id"] = df_raw["id"].astype(str)


with open("data/labeling/manually_labeled_sample.json", "r", encoding="utf-8") as f: #waring: there are posts in chinese 
    labeled = json.load(f)
df_lab = pd.DataFrame(labeled)
df_lab["id"] = df_lab["id"].astype(str)

# here we merged the manual labels with the text
df = df_raw.merge(df_lab, on="id", how="inner")
print("raw:", len(df_raw), "labels:", len(df_lab), "merged:", len(df))

#split train and test set from merged manually labeled samples
train_df, test_df = train_test_split(
    df,
    test_size=0.2, #test set size is set to 20%
    stratify=df["label"]
)

#save data
train_df.to_json("data/training_data/train_set.jsonl", orient="records", lines=True, force_ascii=False)
test_df.to_json("data/training_data/test_set.jsonl", orient="records", lines=True, force_ascii=False)
