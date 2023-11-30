"""
Create som example NER annotations from llm model for review in prodigy
"""

import pandas as pd
import srsly
import torch
from sklearn.cluster import KMeans
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)
from utils import NeuroPipeline, Sectioner


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


tokenizer_name = "models/basemodel"
seg_model = "models/segmentation"
ner_name = "models/ner"
neg_name = "models/negation"
neuro_pipeline = NeuroPipeline(
    ner_name, neg_name, tokenizer_name, aggregation_strategy="first"
)

reports = "data/processed_reports.csv"

df = pd.read_csv(
    reports,
    parse_dates=["End Exam Date"],
    usecols=[
        "Narrative",
        "End Exam Date" "report_header",
        "report_indications",
        "report_metareport",
        "report_body",
        "report_tail",
        "_X",
        "_Y",
    ],
)

sectioner = Sectioner(seg_model, tokenizer_name, device="cuda:0")

nottoolong = df.loc[
    (df["Narrative"].str.len() < 1000) & (df["Narrative"].str.len() > 30)
]


narratives = sample_data(nottoolong, 250)

segmented_reports = sectioner(narratives)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

ner_model_name = "models/ner_biobert/checkpoint-11500"
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first",
    device=torch.device("cuda:0"),
)

all_bodies = [
    i["report_body"]
    for i in segmented_reports
    if "report_body" in i.keys()
    if i["report_body"]
]

ner_output = ner(all_bodies)

pathology_domains = [
    "pathology_cerebrovascular",
    "pathology_congenital_developmental",
    "pathology_csf_disorders",
    "pathology_endocrine",
    "pathology_haemorrhagic",
    "pathology_infectious",
    "pathology_inflammatory_autoimmune",
    "pathology_ischaemic",
    "pathology_metabolic_nutritional_toxic",
    "pathology_neoplastic_paraneoplastic",
    "pathology_neurodegenerative_dementia",
    "pathology_opthalmological",
    "pathology_traumatic",
    "pathology_treatment",
    "pathology_vascular",
]

labels = [
    {
        "text": t,
        "_view_id": "choice",
        "config": {"choice_style": "multiple"},
        "answer": "accept",
        "accept": list(
            set(
                [k["entity_group"] for k in l if k["entity_group"] in pathology_domains]
            )
        ),
        "options": [{"id": j, "text": j} for j in pathology_domains],
    }
    for t, l in zip(all_bodies, ner_output)
]


srsly.write_jsonl("./data/example_ners.jsonl", labels)
