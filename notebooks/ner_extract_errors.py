"""ner domain errors
"""


from pathlib import Path

import numpy as np
import srsly
import torch
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

tokenizer_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
model_name = "models/ner_biobert/checkpoint-17000"
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first",
    device=torch.device("cuda:0"),
)

data_path = "/home/hwatkins/Desktop/neuroData/domains_data/ucl-evaluation-data/domain-review-full-alignopth.jsonl"
gold_path_labels = [i for i in srsly.read_jsonl(data_path) if i["answer"] == "accept"]


classes_to_keep = [
    "pathology-cerebrovascular",
    "pathology-congenital-developmental",
    "pathology-csf-disorders",
    "pathology-endocrine",
    "pathology-haemorrhagic",
    "pathology-infectious",
    "pathology-inflammatory-autoimmune",
    "pathology-ischaemic",
    "pathology-metabolic-nutritional-toxic",
    # "pathology-musculoskeletal",
    "pathology-neoplastic-paraneoplastic",
    "pathology-neurodegenerative-dementia",
    "pathology-opthalmological",
    "pathology-traumatic",
    "pathology-treatment",
    "pathology-vascular",
]


def transform_data(data, classifier):
    """perform the necessary transformation on the input data"""
    gold_path_domains = [
        list((j for j in i["accept"] if j in classes_to_keep)) for i in data
    ]
    texts = [i["text"] for i in data]
    model_predictions = classifier(texts)
    full_neuro_pred_path_domains = []
    for j in model_predictions:
        doc_labels = set()
        for k in j:
            e = k["entity_group"].replace("_", "-")
            if e in classes_to_keep:
                doc_labels.add(e)
        full_neuro_pred_path_domains.append(list(doc_labels))

    return [
        {"text": i, "label": j, "prediction": k}
        for i, j, k in zip(texts, gold_path_domains, full_neuro_pred_path_domains)
    ]


labelled_data = transform_data(gold_path_labels, ner)


srsly.write_jsonl("llm_ner_predictions.jsonl", labelled_data)
