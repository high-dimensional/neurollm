"""Report classifier evaluation
"""
import pandas as pd
import srsly
import torch
from sklearn.metrics import classification_report
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

# %% load
tokenizer = AutoTokenizer.from_pretrained("models/basemodel", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("models/cls")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# data = pd.read_csv("data/cls_data/even_sample_normality_class.csv")
labels = list(srsly.read_jsonl("data/cls_data/labelled_cls_data.jsonl"))
trues = [k["accept"][0] for k in labels if k["answer"] == "accept"]
accepted = [i for i in labels if i["answer"] == "accept"]
texts = [j["text"] for j in accepted]
# %% eval
# output = classifier(data['text'].tolist())
output = classifier(texts)
name_map = {
    "ABNORMAL": "Abnormal",
    "IS_MISSING": "Missing",
    "IS_NORMAL_FOR_AGE": "Normal_for_age",
    "IS_COMPARATIVE": "Comparative",
    "IS_NORMAL": "Normal",
}
# data['prediction'] = [name_map[o['label']] for o in output]
prediction = [name_map[o["label"]] for o in output]
print(classification_report(trues, prediction))
