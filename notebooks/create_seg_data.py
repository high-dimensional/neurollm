"""
Notebook to create prodigy-format segmentation data

"""

import pandas as pd
import spacy
import srsly
from spacy.tokens import Doc, Span

reports = "/home/hwatkins/Desktop/process-reports/data/reports_with_sections_norms.csv"


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
nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "ner"])

report_sample = df.loc[
    (df["Narrative"].str.len() < 1500) & (df["Narrative"].str.len() > 30)
].sample(200)

# %% test
sections = [
    "report_header",
    "report_indications",
    "report_metareport",
    "report_body",
    "report_tail",
]
spacy_docs = []
for row in report_sample.to_dict("records"):
    docs = []
    for s in sections:
        if row[s]:
            doc = nlp(row[s])
            doc.ents = [Span(doc, 0, 1, label=s)]
        docs.append(doc)
    spacy_docs.append(Doc.from_docs(docs))

# %% tospacy


output = [d.to_json() for d in spacy_docs]


srsly.write_jsonl("./data/segmentation_data/seg_predictions.jsonl", output)
bare_output = [{"text": d["text"]} for d in output]
srsly.write_jsonl("./data/segmentation_data/seg_data_to_label.jsonl", bare_output)
