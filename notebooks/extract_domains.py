#!/usr/bin/env python
"""Extract domains from a set of masked reports

This script only applies the domain classification
section of the pipeline, and it assumes the reports 
have been masked.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from neurollm.utils import NeuroPipeline, Sectioner, minibatch

BATCH_SIZE = 32


def load_input(location):
    """load in data"""
    df = pd.read_csv(
        location,
        low_memory=False,
        parse_dates=["End Exam Date"],  # , "Patient DOB", "Finalised Date"],
        # index_col=0,
        infer_datetime_format=True,
    )
    return df


def domain_counts(doc, domains_to_get=[]):
    domains = {a + b: 0 for a in ["asserted_", "denied_"] for b in domains_to_get}
    for e in doc:
        if e["entity_group"] in domains_to_get:
            if e["negated"]:
                domains["denied_" + e["entity_group"]] += 1
            else:
                domains["asserted_" + e["entity_group"]] += 1
    return domains


domain_labels = [
    "location_arteries",
    "location_brain_stem",
    "location_diencephalon",
    "location_ent",
    "location_eye",
    "location_ganglia",
    "location_grey_matter",
    "location_limbic_system",
    "location_meninges",
    "location_nerves",
    "location_neurosecretory_system",
    "location_other",
    "location_skull",
    "location_spine",
    "location_telencephalon",
    "location_veins",
    "location_ventricles",
    "location_white_matter",
    "location_qualifier",
    "descriptor_cyst",
    "descriptor_damage",
    "descriptor_diffusion",
    "descriptor_signal_change",
    "descriptor_enhancement",
    "descriptor_flow",
    "descriptor_interval_change",
    "descriptor_mass_effect",
    "descriptor_morphology",
    "descriptor_collection",
    "pathology_haemorrhagic",
    "pathology_ischaemic",
    "pathology_vascular",
    "pathology_cerebrovascular",
    "pathology_treatment",
    "pathology_inflammatory_autoimmune",
    "pathology_congenital_developmental",
    "pathology_csf_disorders",
    "pathology_musculoskeletal",
    "pathology_neoplastic_paraneoplastic",
    "pathology_infectious",
    "pathology_neurodegenerative_dementia",
    "pathology_metabolic_nutritional_toxic",
    "pathology_endocrine",
    "pathology_opthalmological",
    "pathology_traumatic",
    "descriptor_necrosis",
]


def get_domains(df, package_dir):
    print("processing domain detection")
    tokenizer_name = package_dir / "basemodel"
    ner_name = package_dir / "ner"
    neg_name = package_dir / "negation"
    bodies = df["report_body_masked"].fillna(" ").tolist()
    pipeline = NeuroPipeline(
        ner_name,
        neg_name,
        tokenizer_name,
        aggregation_strategy="first",
        device="cuda:0",
        batch_size=BATCH_SIZE,
    )
    doc_domains = [
        domain_counts(d, domains_to_get=domain_labels)
        for d in tqdm(pipeline(bodies), total=len(bodies) // BATCH_SIZE)
    ]
    for i in ["asserted_", "denied_"]:
        for j in domain_labels:
            key = i + j
            df[key] = [k[key] for k in doc_domains]
    del pipeline
    return df


def transform_data(df, package_dir):
    """perform the necessary transformation on the data"""
    processed_df = df.pipe(get_domains, package_dir)
    return processed_df


def output_results(data, outdir):
    """output analysis, save to file or send to stdout"""
    data.to_csv(outdir / "processed_masked_reports.csv", index=False)


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
    data = load_input(args.input)
    transformed_data = transform_data(data, args.model_dir)
    output_results(transformed_data, args.outdir)


if __name__ == "__main__":
    main()
