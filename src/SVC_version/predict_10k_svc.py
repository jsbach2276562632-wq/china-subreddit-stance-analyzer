import pandas as pd
import numpy as np
import joblib

# load the model and input (reddits with mention regime)
clf = joblib.load("models/stance_lr.joblib")
df = pd.read_csv("data/processed/reddit_with_mention_regime.csv")

# Confidence threshold, only accept predictions with higher threshold
CO_TH = 0.45

X = df["text"].fillna("").astype(str)

scores = clf.decision_function(X)       # use our saved pipeline to predict the decision score of X(text)

pred_idx = np.argmax(scores, axis=1)  # take the predictions with the max probablity
pred = clf.classes_[pred_idx]  

# we can only calculate the gap between the best and second best score as there
#are no probabilities with LinearSVC
sorted_scores = np.sort(scores, axis=1)
gap = sorted_scores[:, -1] - sorted_scores[:, -2]

#write the result in to df
df["predicted_label"] = pred
df["score_gap"] = gap # "how better is the score we chosed compared to the second best alternative"

q = 0.30  # keep top 70% most confident. we need more labels.....
TH = np.quantile(gap, q)

# np.where(condition, value_if_true, value_if_false)
df["confident_label"] = np.where(df["score_gap"] >= TH, df["predicted_label"], "undecidable")


#save progress
df.to_csv("data/processed/reddit_with_mention_regime_labeled.csv", index=False, encoding="utf-8")
print("Saved: data/processed/reddit_with_mention_regime_SVC_abeled.csv")

