#!/usr/bin/env python
"""relabel prospective data with llm model predictions

"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from srsly import read_jsonl, write_jsonl
from utils import NeuroPipeline

PATHOLOGY_DOMAINS = [
    "pathology-cerebrovascular",
    "pathology-congenital-developmental",
    "pathology-csf-disorders",
    "pathology-endocrine",
    "pathology-haemorrhagic",
    "pathology-infectious",
    "pathology-inflammatory-autoimmune",
    "pathology-ischaemic",
    "pathology-metabolic-nutritional-toxic",
    "pathology-neoplastic-paraneoplastic",
    "pathology-neurodegenerative-dementia",
    "pathology-opthalmological",
    "pathology-traumatic",
    "pathology-treatment",
    "pathology-vascular",
]


tokenizer_name = "../models/basemodel"
model_name = "../models/negation"
ner_name = "../models/ner"
neuro_pipeline = NeuroPipeline(
    ner_name, model_name, tokenizer_name, aggregation_strategy="first"
)
DATA = "/home/hwatkins/Desktop/neuroData/domains_data/ucl-evaluation-data/prospective-domain-data.jsonl"

data = list(read_jsonl(DATA))
