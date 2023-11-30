"""Evaluate segmentation performace
"""
import argparse
from pathlib import Path

import spacy
import srsly
from sklearn.metrics import classification_report
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example

from neurollm.utils import Sectioner

# %% load

nlp = spacy.load("en_core_web_sm", exclude=["ner"])
labels = srsly.read_jsonl("./data/segmentation_data/labelled_seg_data.jsonl")
accepted = [l for l in labels if l["answer"] == "accept"]
model = Sectioner("models/segmentation", "models/basemodel")


# %% convert
def make_doc(section_dict):
    sections = [
        "report_header",
        "report_indications",
        "report_metareport",
        "report_body",
        "report_tail",
    ]
    docs = []
    for s in sections:
        if s in section_dict.keys():
            doc = nlp(section_dict[s])
            doc.ents = [Span(doc, 0, 1, label=s)]
            docs.append(doc)
    final_doc = Doc.from_docs(docs)
    return final_doc


def convert_prodigy(prodigy_dict):
    doc = nlp(prodigy_dict["text"])
    spans = [
        Span(doc, s["token_start"], s["token_end"] + 1, label=s["label"])
        for s in prodigy_dict["spans"]
    ]
    doc.ents = spans
    return doc


predictions = model([t["text"] for t in accepted])
pred_docs = [make_doc(out) for out in predictions]
label_docs = [convert_prodigy(a) for a in accepted]
# %% eval
examples = [Example(p, l) for p, l in zip(pred_docs, label_docs)]
scorer = Scorer(nlp)
results = scorer.score_spans(examples, "ents")
srsly.write_json("metrics/segmentation_results.json", results)
