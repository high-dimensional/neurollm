# neurollm

llm-based nlp pipelines for neuro reports. 

Natural language processing pipelines for neurological text.
This module contains models and pipelines for the following tasks

1. segmentation
2. ner 
3. normality classification
4. ner+negation custom pipeline

## Installation

Clone the repository and install via
`pip install -e .`

## Usage

Segmentation
```python
from neurollm.utils import Sectioner
text1 = "No previous imaging available for comparison at the time of reporting. There is a large right-sided intracranial mass lesion centered on the right frontal lobe."
tokenizer = "./models/basemodel"
model = "./models/segmentation"
sectioner = Sectioner(model, tokenizer)
print(list(sectioner([text1])))
```

NER+negation
```python
from neurollm.utils import NeuroPipeline
text1 = "No previous imaging available for comparison at the time of reporting. There is a large right-sided intracranial mass lesion centered on the right frontal lobe."
tokenizer = "./models/basemodel"
ner_model = "./models/ner"
neg_model = "./models/negation"
pipeline = NeuroPipeline(
    ner_model, neg_model, tokenizer, aggregation_strategy="first"
)
output = pipeline([text1])
for i in output:
    for j in i:
        print(j["word"], ":", j["entity_group"], ":", j["negated"])
```

Normality classification
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
text1 = "There is a large right-sided intracranial mass lesion centered on the right frontal lobe."
tokenizer_name = "./models/basemodel"
cls_name = "./models/cls"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(cls_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
output = classifier([text1])
print(output)

```

