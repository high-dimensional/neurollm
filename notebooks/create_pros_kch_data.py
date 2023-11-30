"""
Create som example NER annotations from llm model for review in prodigy
"""
import random

import numpy as np
# %%
import pandas as pd
import srsly
import torch
from sklearn.cluster import KMeans

from neurollm.utils import NeuroPipeline


# %%
# %%
def sample_data(data, size):
    """perform the necessary transformation on the input data"""
    n_clusters = 100
    data = data.assign(
        cluster_id=lambda x: KMeans(n_clusters=n_clusters).fit_predict(x[["_X", "_Y"]])
    )
    cluster_sample = size // n_clusters
    sample = (
        data.groupby("cluster_id")
        .sample(cluster_sample, replace=True)["report_body_masked"]
        .fillna("")
        .tolist()
    )
    return sample


tokenizer_name = "/home/hwatkins/Desktop/neurollm/models/basemodel"
ner_name = "/home/hwatkins/Desktop/neurollm/models/ner-fine-tuned"
neg_name = "/home/hwatkins/Desktop/neurollm/models/negation"
neuro_pipeline = NeuroPipeline(
    ner_name, neg_name, tokenizer_name, aggregation_strategy="first"
)

# reports = "data/processed_reports.csv"
reports = "/home/hwatkins/Desktop/process-reports/kch_data/reports_with_domains.csv"
full_df = pd.read_csv(
    reports,
    parse_dates=["End Exam Date"],
)

full_df["_X"] = np.random.rand(len(full_df))
full_df["_Y"] = np.random.rand(len(full_df))
df = full_df  # [full_df["End Exam Date"] > pd.to_datetime("2021-09-01", format="%Y-%m-%d")]

nottoolong = df.loc[
    (df["report_body_masked"].str.len() < 1000)
    & (df["report_body_masked"].str.len() > 30)
]

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

domains_to_sample = [
    "pathology_cerebrovascular",
    "pathology_endocrine",
    "pathology_opthalmological",
]


def even_sample(df, N, domains=[]):
    n = int(N / len(pathology_domains))
    domain_samples = []
    for d in domains:
        has_domain = df[df["asserted_" + d] > 0]
        sample_size = len(has_domain) if len(has_domain) < n else n
        domain_sample = has_domain.sample(sample_size)
        domain_samples.append(domain_sample)
    sample_df = pd.concat(domain_samples)
    return sample_df["report_body_masked"].tolist()


# narratives = sample_data(nottoolong, 500)
narratives = even_sample(nottoolong, 500, domains=domains_to_sample)
print("sample: ", narratives[:3])
random.shuffle(narratives)
ner_output = neuro_pipeline(narratives)

has_musc = lambda x: any(
    [i["entity_group"] == "pathology_musculoskeletal" for i in x["ents"]]
)
no_musc = [out for out in ner_output if not has_musc(out)]


labels = [
    {
        "text": datum["text"],
        "_view_id": "choice",
        "config": {"choice_style": "multiple"},
        "answer": "accept",
        "accept": list(
            set(
                [
                    k["entity_group"]
                    for k in datum["ents"]
                    if (k["entity_group"] in pathology_domains) and (not k["negated"])
                ]
            )
        ),
        "options": [{"id": j, "text": j} for j in pathology_domains],
    }
    for datum in no_musc
]


srsly.write_jsonl("./data/prospective_domain_labels_KCH_v5.jsonl", labels)
