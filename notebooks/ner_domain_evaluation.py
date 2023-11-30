"""ner domain evaluation

Use the ner model and evaluate on labelled ucl multilabel data,
compare with basic neuroNLP model
"""


from pathlib import Path

import numpy as np
import srsly
import torch
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

from neurollm.utils import NeuroPipeline

tokenizer_name = "./models/basemodel"
ner_model_name = "./models/ner-fine-tuned"
neg_model_name = "./models/negation"
# data_path = "/home/hwatkins/Desktop/neuroData/domains_data/ucl-evaluation-data/domain-review-full-alignopth.jsonl"
data_path = "/home/hwatkins/Desktop/neuroData/domains_data/kcl-evaluation-data/domain_review_kcl_alignopth.jsonl"
# data_path = "/home/hwatkins/Desktop/neuroData/domains_data/ucl-evaluation-data/ucl_prospective_labels.jsonl"
# data_path = "/home/hwatkins/Desktop/neuroData/domains_data/kcl-evaluation-data/prospective_labelled_domains_llm.jsonl"
gold_path_labels = [i for i in srsly.read_jsonl(data_path) if i["answer"] == "accept"]
NAME = "kch-llm"
nlp = NeuroPipeline(ner_model_name, neg_model_name, tokenizer_name)


def multilabel_specificity(y_true, y_pred, classes):
    mlcm = multilabel_confusion_matrix(y_true, y_pred)
    tn, fn, tp, fp = mlcm[:, 0, 0], mlcm[:, 1, 0], mlcm[:, 1, 1], mlcm[:, 0, 1]
    specifics = tn / (tn + fp)
    output = {i: j for i, j in zip(classes, specifics)}
    output["micro avg"] = tn.sum() / (tn.sum() + fp.sum())
    output["macro avg"] = specifics.mean()
    support = y_true.sum(0)
    output["weighted avg"] = (specifics * support).sum() / support.sum()
    return output


"""
classes_to_keep = [
    "pathology-cerebrovascular",
    "pathology-congenital-developmental",
    "pathology-csf-disorders",
    "pathology-endocrine",
    "pathology-haemorrhagic",
    "pathology-infectious",
    "pathology-inflammatory-autoimmune",
    "pathology-ischaemic",
    "pathology-metabolic-nutritional-toxic",
    # "pathology-musculoskeletal",
    "pathology-neoplastic-paraneoplastic",
    "pathology-neurodegenerative-dementia",
    "pathology-opthalmological",
    "pathology-traumatic",
    "pathology-treatment",
    "pathology-vascular",
]
"""

classes_to_keep = [
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


def transform_data(data, classifier):
    """perform the necessary transformation on the input data"""
    gold_path_domains = [
        list((j for j in i["accept"] if j in classes_to_keep)) for i in data
    ]
    texts = [i["text"] for i in data]
    model_predictions = classifier(texts)
    full_neuro_pred_path_domains = []
    for j in model_predictions:
        doc_labels = set()
        for k in j["ents"]:
            e = k["entity_group"]  # .replace("_", "-")
            if (e in classes_to_keep) and (not k["negated"]):
                doc_labels.add(e)
        full_neuro_pred_path_domains.append(list(doc_labels))
    path_bina = MultiLabelBinarizer()
    path_true = path_bina.fit_transform(gold_path_domains)
    path_neuro_pred = path_bina.transform(full_neuro_pred_path_domains)
    return path_bina, path_true, path_neuro_pred


def analysis(binarizer, true_label, pred_label):
    """perform analysis on data"""
    mtlconf = multilabel_confusion_matrix(true_label, pred_label)
    output = classification_report(
        true_label,
        pred_label,
        target_names=binarizer.classes_,
        zero_division=0,
        output_dict=True,
    )
    specifics = multilabel_specificity(true_label, pred_label, binarizer.classes_)
    for i, label in enumerate(output.keys()):
        if label in specifics.keys():
            output[label]["specificity"] = specifics[label]
            output[label]["sensitivity"] = output[label]["recall"]
    return mtlconf, output


binarizer, true_labels, pred_labels = transform_data(gold_path_labels, nlp)
conf_mat, output = analysis(binarizer, true_labels, pred_labels)


def output_to_file(
    file, conf_mat, model, binarizer, gold_labels, true_labels, pred_labels
):
    file.write("NER-based performance\n")

    for i, name in enumerate(binarizer.classes_):
        file.write(name + "\n")
        file.write(str(conf_mat[i]) + "\n")

    for i, v in enumerate(binarizer.classes_):
        idx_fp = np.logical_and(true_labels[:, i] == 0, pred_labels[:, i] == 1)
        idx_fn = np.logical_and(true_labels[:, i] == 1, pred_labels[:, i] == 0)

        fp_docs = [d for d, i in zip(gold_labels, idx_fp) if i]
        file.write(f"\nFalse positives for {v}\n")
        for t in fp_docs:
            file.write(t["text"])
            file.write("\n")
            doc = list(model(t["text"]))
            file.write(
                "ents: "
                + str(
                    [
                        (e["word"], e["entity_group"])
                        for e in doc[0]["ents"]
                        if (e["entity_group"].replace("_", "-") in classes_to_keep)
                        and (not e["negated"])
                    ]
                )
            )
            file.write("\n")
            file.write("\n")

        fn_docs = [d for d, i in zip(gold_labels, idx_fn) if i]
        file.write(f"\nFalse negatives for {v}\n")
        for t in fn_docs:
            file.write(t["text"])
            file.write("\n")
            doc = list(model(t["text"]))
            file.write(
                "ents: "
                + str(
                    [
                        (e["word"], e["entity_group"])
                        for e in doc[0]["ents"]
                        if (e["entity_group"].replace("_", "-") in classes_to_keep)
                        and (not e["negated"])
                    ]
                )
            )
            file.write("\n")
            file.write("\n")


def output_results(
    outdir,
    confusion_matrix,
    output,
    binarizer,
    gold_labels,
    true_labels,
    pred_labels,
    name,
):
    """output analysis, save to file or send to stdout"""
    model_name = name
    with open(outdir / f"domain-perf-{model_name}.txt", "w") as f:
        output_to_file(
            f, confusion_matrix, nlp, binarizer, gold_labels, true_labels, pred_labels
        )
    srsly.write_json(outdir / f"domain-perf-{model_name}.json", output)


outdir = Path("metrics")
output_results(
    outdir,
    conf_mat,
    output,
    binarizer,
    gold_path_labels,
    true_labels,
    pred_labels,
    NAME,
)
