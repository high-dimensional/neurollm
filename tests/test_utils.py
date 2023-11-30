import pytest
from spacy.tokens import Doc

from neurollm.utils import *

texts = [
    "7210825 20/02/2018 MR Head 7210825 20/02/2018 Imaging under GA 7210825 20/02/2018 MR MRA Clinical Indication: Stroke patient thrombolysed 19/02/18. Low GCS I+V to ICU NOTE: Patient likely I+V On ICU Findings: Reference made to CT scan dated 19/02/18. No foci of restricted diffusion to suggest infarction. No evidence of haemorrhage and no focal mass lesion. The irregularity of M1 segment of the left middle cerebral artery is not well demonstrated on TOF-MRI. Normal flow-related signal is seen in the imaged portions of ICA, vertebral arteries and intracranial arteries. Conclusion: No evidence of infarction. GC Dr P S Rangi Consultant Neuroradiologist GMC NO: 4189686 Email: neurorad.uclh.nhs.uk",
    "Clinical Indications for MRI - White matter lesions on earlier scan has phospholipuid syndrome but ? MS Will need contrast Findings: Comparison is made with the previous scan performed 6 April 2016. Stable appearances of the diffuse and confluent bilateral white matter high T2/FLAIR signal changes. A few more conspicuous focal lesions in the periventricular and juxta-cortical white matter are again demonstrated and unchanged. There is no evidence of signal changes in the posterior fossa structures. The imaged cervical cord returns normal signal. There is no evidence of pathological enhancement. Summary: Stable appearances of the supratentorial white matter signal changes. Although some lesions appear more conspicuous in the periventricular and juxtacortical regions, there is no significant lesion burden to characterise dissemination in space at this time point. Dr Kelly Pegoretti Consultant Neuroradiologist email: neurorad@uclh.nhs.uk",
    "words and more words",
    "   ",
]


@pytest.fixture
def model():
    ner_name = "./models/ner"
    model_name = "./models/negation"
    tokenizer_name = "./models/basemodel"  # "dmis-lab/biobert-v1.1"
    neuro_pipeline = NeuroPipeline(
        ner_name,
        model_name,
        tokenizer_name,
        aggregation_strategy="first",
        device="cuda:0",
    )
    return neuro_pipeline


@pytest.fixture
def docs(model):
    output = list(model(texts))
    MODEL = "en_core_web_lg"
    converter = DocConverter(MODEL)
    doc1 = converter(output[0])
    doc2 = converter(output[1])
    return [doc1, doc2]


def test_neuro_pipeline(model):
    output = list(model(texts))
    assert output[0]["ents"][0]["entity_group"] == "pathology_cerebrovascular"


def test_sectioner():
    tokenizer = "./models/basemodel"  # "dmis-lab/biobert-v1.1"
    model = "./models/segmentation"
    sectioner = Sectioner(model, tokenizer)
    big_texts = [texts[0] for i in range(100)]
    output = list(sectioner(big_texts))
    true_body = "No foci of restricted diffusion to suggest infarction. No evidence of haemorrhage and no focal mass lesion. The irregularity of M1 segment of the left middle cerebral artery is not well demonstrated on TOF - MRI. Normal flow - related signal is seen in the imaged portions of ICA, vertebral arteries and intracranial arteries. Conclusion : No evidence of infarction."
    assert output[0]["report_body"] == true_body


def test_pipeline2doc(docs):
    doc = docs[0]
    assert isinstance(doc, Doc)
    assert doc.ents[0].text == "Stroke"
    assert doc.ents[0].label_ == "pathology_cerebrovascular"
    assert not doc.ents[0]._.negex


def test_relex(docs):
    doc2 = docs[1]
    relex = RelationExtractor(doc2.vocab)
    relex.from_disk("/home/hwatkins/Desktop/neurollm/models/relex_small")
    new_doc = relex(doc2)
    print(new_doc._.rel)
    print([(e, e._.relation) for e in new_doc.ents])
    assert new_doc.ents[2]._.relation[0].text == "white matter"


def test_doc2array(docs):
    relex = RelationExtractor(docs[0].vocab)
    relex.from_disk("/home/hwatkins/Desktop/neurollm/models/relex_small")
    vctr = DocVectorizer()
    array = vctr.doc2array(docs[0])
    assert array[1, 0, 1] == 2
