import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

text1 = ""
text2 = ""
tokenizer_name = "dmis-lab/biobert-v1.1"  # "roberta-base"
model_name = "models/cls/checkpoint-5000"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

output = classifier([text1, text2])
print(output)
