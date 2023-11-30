from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from neurollm.utils import NeuroPipeline, Sectioner, minibatch

ner_name = "./models/ner"
neg_name = "./models/negation"
tokenizer_name = "./models/basemodel"

pipeline = NeuroPipeline(
    ner_name,
    neg_name,
    tokenizer_name,
    aggregation_strategy="first",
)

texts = []

output = pipeline(texts)
for text, out in zip(texts, output):
    print(text, out)
