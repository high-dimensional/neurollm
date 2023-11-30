#!/usr/bin/env python
"""Mask out certain domains

output a dataframe with report bodies with commentary masked
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from negspacy.negation import Negex
from neuroNLP.custom_pipes import *
from tqdm import tqdm

nprocs = 24
batchn = 512


def load_input(location):
    """load in data"""
    df = pd.read_csv(
        location,
        low_memory=False,
        index_col=0,
    )
    return df


def get_domains(df, package_dir):
    print("processing domain detection")
    domain_nlp = spacy.load(
        package_dir
        / "en_full_neuro_model-1.8"
        / "en_full_neuro_model"
        / "en_full_neuro_model-1.8"
    )
    bodies = df["report_body"]

    body_docs = domain_nlp.pipe(bodies, batch_size=batchn, n_process=nprocs)
    print("spine masking")
    df["report_body_masked"] = [
        remove_commentary(i) for i in tqdm(body_docs, total=len(bodies))
    ]
    del body_docs, bodies
    return df


def transform_data(df, package_dir):
    """perform the necessary transformation on the data"""

    processed_df = df.pipe(get_domains, package_dir)

    return processed_df


def output_results(data, outdir):
    """output analysis, save to file or send to stdout"""
    data.to_csv(outdir / "masked_reports.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="report CSV", type=Path)
    parser.add_argument("model_dir", help="Path to the NLP model directory", type=Path)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    parser.add_argument(
        "--gpu", help="use gpu to accelerate processing", action="store_true"
    )
    args = parser.parse_args()
    if not args.outdir.exists():
        args.outdir.mkdir()
    if args.gpu:
        spacy.prefer_gpu()
    data = load_input(args.input)
    transformed_data = transform_data(data, args.model_dir)
    output_results(transformed_data, args.outdir)


if __name__ == "__main__":
    main()
