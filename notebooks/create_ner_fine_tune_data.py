#!/usr/bin/env python
"""Preprocess ner data.

Create some NER data with the spacy ner pipeline and save as training data for huggingface pipeline
This data specifically is aimed at 'training-out' errors for 

sinusitis
Infection risk/protocol
"""
import argparse
from pathlib import Path

import pandas as pd
import spacy
from datasets import Dataset
from neuroNLP.custom_pipes import *
from sklearn.cluster import KMeans
from spacy.tokens import Span
from tqdm import tqdm


def load_input(location):
    """load in data"""
    columns = [
        "Narrative",
        "report_body",
    ]
    df = pd.read_csv(location, low_memory=False, usecols=columns)
    return df


def sample_data(data, size):
    """sample data specifically containing features of concern"""
    has_features = data[
        data["Narrative"].str.contains("sinusitis|Infection Risk", na=False)
    ]
    sample = has_features.sample(size, replace=True)["Narrative"].fillna("")
    return sample


def convert_ents(doc):
    new_ents = [
        Span(doc, e.start, e.end, label=e.label_.replace("-", "_")) for e in doc.ents
    ]
    doc.ents = new_ents
    return doc


def get_ner_labels(docs):
    """extract the tokens and ner labels for a set of space docs"""
    get_tags_tokens = lambda x: dict(zip(["ner_tags", "tokens"], list(zip(*x))))
    get_items = lambda x: (
        (w.ent_iob_ + "-" + w.ent_type_, w.text)
        if w.ent_type_
        else (w.ent_iob_, w.text)
        for w in x
    )
    all_tags_tokens = [
        get_tags_tokens(get_items(convert_ents(doc))) for doc in tqdm(docs)
    ]
    return all_tags_tokens


def remove_bug_labels(ner_labels):
    treated_labels = []
    for d in tqdm(ner_labels):
        new_tags = []
        for tag, token in zip(d["ner_tags"], d["tokens"]):
            if ("Infection" in token) or ("sinusitis" in token):
                new_tags.append("O")
            else:
                new_tags.append(tag)
        treated_labels.append(
            {"ner_tags": tuple(new_tags), "tokens": tuple(d["tokens"])}
        )

    return treated_labels


def output_results(data, outdir):
    """output analysis, save to file or send to stdout"""
    ds = Dataset.from_pandas(pd.DataFrame(data=data))
    ds.save_to_disk(outdir)


OUTDIR = Path("/home/hwatkins/Desktop/neurollm/data/fine_tune_ner_data")
MODEL = Path(
    "/home/hwatkins/Desktop/neuroNLP/packages/en_full_neuro_model-1.8/en_full_neuro_model/en_full_neuro_model-1.8"
)
INPUT = Path(
    "/home/hwatkins/Desktop/neuroData/KCH_report_data_2022t2023/processed_reports.csv"
)
SIZE = 3000

if not OUTDIR.exists():
    OUTDIR.mkdir()
data = load_input(INPUT)
nlp = spacy.load(MODEL, exclude=["negex"])
data_sample = sample_data(data, SIZE)
data_sample.to_csv(OUTDIR / "sample_data.csv")
docs = nlp.pipe(data_sample, n_process=16, batch_size=32)
transformed_data = get_ner_labels(docs)
filtered_data = remove_bug_labels(transformed_data)
output_results(filtered_data, OUTDIR)
