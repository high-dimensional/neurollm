"""negex method comparison script

take assertion dataset and compare the results of the two negex methods,
specifically for llm-based joint ner/negation pipeline
"""

import srsly

from neurollm.utils import NeuroPipeline

# load the data
data_path = "/home/hwatkins/Desktop/neuroData/domains_data/ucl-evaluation-data/domain-review-full-alignopth.jsonl"
gold_path_labels = [i for i in srsly.read_jsonl(data_path) if i["answer"] == "accept"]

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
    "pathology-neoplastic-paraneoplastic",
    "pathology-neurodegenerative-dementia",
    "pathology-opthalmological",
    "pathology-traumatic",
    "pathology-treatment",
    "pathology-vascular",
]

ner_name = "/home/hwatkins/Desktop/neurollm/models/ner/checkpoint-12000"
model_name = "/home/hwatkins/Desktop/neurollm/models/negation/checkpoint-12000"
tokenizer_name = "dmis-lab/biobert-v1.1"
nlp = NeuroPipeline(
    ner_name, model_name, tokenizer_name, aggregation_strategy="first", device="cuda:0"
)

outputs = nlp([i["text"] for i in gold_path_labels])
llm_preds = []
for prediction, datum in zip(outputs, gold_path_labels):
    new_datum = datum
    predicted_classes = {
        i["entity_group"].replace("_", "-") for i in prediction if not i["negated"]
    }
    new_datum["accept"] = [j for j in predicted_classes if j in classes_to_keep]
    llm_preds.append(new_datum)
srsly.write_jsonl("llm_ner_neg_predictions.jsonl", llm_preds)
