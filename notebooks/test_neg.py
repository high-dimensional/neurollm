import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)
from utils import NeuroPipeline

text1 = ""
text2 = ""
tokenizer_name = "dmis-lab/biobert-v1.1"  #'allenai/biomed_roberta_base'#"roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

model_name = "../models/negation/checkpoint-12000"
model = AutoModelForTokenClassification.from_pretrained(model_name)
classifier = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="first"
)

output = classifier([text1, text2])
print(output)
for i in output:
    for j in i:
        print(j["word"], ":", j["entity_group"])

ner_name = "../models/ner/checkpoint-12000"
neuro_pipeline = NeuroPipeline(
    ner_name, model_name, tokenizer_name, aggregation_strategy="first"
)

output = neuro_pipeline([text1, text2])
print(output)
for i in output:
    for j in i:
        print(j["word"], ":", j["entity_group"], ":", j["negated"])
