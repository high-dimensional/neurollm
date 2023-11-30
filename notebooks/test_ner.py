import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

text1 = ""
text2 = ""
tokenizer_name = "dmis-lab/biobert-v1.1"  #'allenai/biomed_roberta_base'#"roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

model_name = "models/ner-fine-tuned/checkpoint-500"
model = AutoModelForTokenClassification.from_pretrained(model_name)
classifier = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first",
)

output = classifier([text1, text2])
print(output)
for i in output:
    for j in i:
        print(j["word"], ":", j["entity_group"])
