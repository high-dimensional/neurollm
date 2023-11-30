"""
Notebook to create prodigy-format segmentation data

"""

import pandas as pd
import srsly
from utils import Sectioner

tokenizer = "dmis-lab/biobert-v1.1"  # "roberta-base"
model = "models/segmentation/checkpoint-1000"

reports = "/home/hwatkins/Desktop/neuroData/processed_reports/processed_reports.csv"

df = pd.read_csv(
    reports,
    usecols=[
        "Narrative",
        "report_header",
        "report_indications",
        "report_metareport",
        "report_body",
        "report_tail",
    ],
)

sectioner = Sectioner(model, tokenizer, device="cuda:0")

report_sample = df.loc[
    (df["Narrative"].str.len() < 1500) & (df["Narrative"].str.len() > 30)
].sample(200)

narratives = report_sample["Narrative"].tolist()

segmented_reports = sectioner(narratives)

output = [
    {"text": str(t) + "\n\n" + "\n\n".join([k + "\n" + v for k, v in s.items()])}
    for t, s in zip(narratives, segmented_reports)
]

srsly.write_jsonl("./data/example_segmentations.jsonl", output)
