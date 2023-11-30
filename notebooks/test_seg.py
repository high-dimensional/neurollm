from utils import Sectioner

tokenizer = "dmis-lab/biobert-v1.1"  # "roberta-base"
model = "../models/segmentation/checkpoint-1000"


sectioner = Sectioner(model, tokenizer)
print(sectioner([text2, text1, text3]))
