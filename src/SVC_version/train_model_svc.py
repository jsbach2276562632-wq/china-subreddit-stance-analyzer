import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline # using pipeline for Text Vectorization and classifier
from sklearn.feature_extraction.text import TfidfVectorizer # vectorzation
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # evaluate modal


train_df = pd.read_json("data/training_data/train_set.jsonl", lines=True)
test_df  = pd.read_json("data/training_data/test_set.jsonl", lines=True)

X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_test  = test_df["text"].fillna("") # test set doesnt get involved in training process
y_test  = test_df["label"]

#Model setting
clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),  # words and combinations of two words (namely unigram and bigram)
        min_df=2, # only keep words that appears at least twice
        max_df=0.95 # drop words that appear in more than 95% of all text
        )),
    ("svm", LinearSVC(class_weight="balanced")) #our sample is unbalanced due to too many neutral
])

clf.fit(X_train, y_train) # fit the train set

#evaluation
pred = clf.predict(X_test) # use the model to predict the test set
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("\nClassification Report:")
print(classification_report(y_test, pred, digits=3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred, labels=["critical", "neutral", "positive"]))


#save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/stance_svc.joblib")
print("Saved: models/stance_svc.joblib")
