import pandas as pd
import json
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer

# load the sets
train_df = pd.read_json("data/training_data/train_set.jsonl", lines=True)
test_df  = pd.read_json("data/training_data/test_set.jsonl", lines=True)


# make sure our text is string
train_df["text"] = train_df["text"].astype(str)


label_dict = {
    "critical": 0,
    "neutral": 1,
    "positive": 2
}
train_df["label_id"] =train_df["label"].map(label_dict)# transform stances into numbers

#did the same to test_df
test_df["text"] = test_df["text"].astype(str)
label_dict = {
    "critical": 0,
    "neutral": 1,
    "positive": 2
}
test_df["label_id"] =test_df["label"].map(label_dict)

# make lists for tokenizer 
train_texts = train_df["text"].tolist()
train_labels = train_df["label_id"].tolist()

test_texts = test_df["text"].tolist()
test_labels = test_df["label_id"].tolist()

# tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased") 

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True, #fill the blank
    max_length=256
)


# build PyTorch standardized interface from our raw data
class Dataset_builder(torch.utils.data.Dataset):  # define builder in the format of torch dataset
    def __init__(self, encodings, labels): #initialization
        self.encodings = encodings #encodings are our already packed texts
        self.labels = labels

    def __getitem__(self, idx): # return text to the model when it asks
        item = {}
        for key in self.encodings:
            item[key] = torch.tensor(self.encodings[key][idx])
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self): # how many we have 
        return len(self.labels)


train_dataset = Dataset_builder(train_encodings, train_labels)
test_dataset = Dataset_builder(test_encodings, test_labels)


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

#setting arguments for training
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", #evaluate after each eopch
    save_strategy="epoch",  
    num_train_epochs=3, # insgesamt 3 Runde (Eopochen?) zu trainieren
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# predict on test set
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(axis=1)


reverse_label_dict = { 
    0: "critical",
    1: "neutral",
    2: "positive"
} # do reversely the transformation 

true_label = [reverse_label_dict[i] for i in test_labels]
pred_label = [reverse_label_dict[i] for i in pred_labels]


print(classification_report(true_label, pred_label))


save_path = "models/distilbert"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
#its saved path is /home/ytzhou/China_Project/models/distilbert/distilbert_simple

print("model saved")