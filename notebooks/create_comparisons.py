
import srsly
from utils import Sectioner as SectionerHF

tokenizer = "roberta-base"
model = "models/segmentation/checkpoint-2000"
hfsectioner = SectionerHF(model, tokenizer)
data = list(srsly.read_jsonl("segmentation_comparison_data.jsonl"))
texts = [i["text"] for i in data]
sectioned_reports_hf = hfsectioner(texts)
all_data = [
    {"text": datum["text"], "nlp_pred": datum["nlp_pred"], "hf_pred": hf}
    for datum, hf in zip(data, sectioned_reports_hf)
]
srsly.write_jsonl("segmentation_comparison_data.jsonl", all_data)
