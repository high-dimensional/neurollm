#!/usr/bin/env python
"""Preprocess ner data.

Create some NER data with the spacy ner pipeline and save as training data for huggingface pipeline
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
        "_X",
        "_Y",
    ]
    df = pd.read_csv(location, low_memory=False, usecols=columns)
    return df


def sample_data(data, size):
    """perform the necessary transformation on the input data"""
    n_clusters = 100
    data = data.assign(
        cluster_id=lambda x: KMeans(n_clusters=n_clusters).fit_predict(x[["_X", "_Y"]])
    )
    cluster_sample = size // n_clusters
    sample = (
        data.groupby("cluster_id")
        .sample(cluster_sample, replace=True)["Narrative"]
        .fillna("")
        .tolist()
    )
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


def output_results(data, outdir):
    """output analysis, save to file or send to stdout"""
    ds = Dataset.from_pandas(pd.DataFrame(data=data))
    ds.save_to_disk(outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="report data", type=Path)
    parser.add_argument("model", help="spacy ner model", type=Path)
    parser.add_argument("-s", "--size", help="dataset size", type=int, default=1000)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    args = parser.parse_args()
    data = load_input(args.input)
    nlp = spacy.load(args.model)
    data_sample = sample_data(data, args.size)
    docs = nlp.pipe(data_sample, n_process=16, batch_size=32)
    transformed_data = get_ner_labels(docs)
    if not args.outdir.exists():
        args.outdir.mkdir()
    output_results(transformed_data, args.outdir)


if __name__ == "__main__":
    main()
