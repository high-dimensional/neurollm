from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import spacy
import srsly
import torch
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Span
from spacy.util import ensure_path, from_disk
from tqdm import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Pipeline, pipeline)

LOCATION_LABELS = [
    "location_arteries",
    "location_brain_stem",
    "location_diencephalon",
    "location_ent",
    "location_eye",
    "location_ganglia",
    "location_grey_matter",
    "location_limbic_system",
    "location_meninges",
    "location_nerves",
    "location_neurosecretory_system",
    "location_other",
    "location_skull",
    "location_spine",
    "location_telencephalon",
    "location_veins",
    "location_ventricles",
    "location_white_matter",
    "location_qualifier",
]

DESCRIPTOR_LABELS = [
    "descriptor_cyst",
    "descriptor_damage",
    "descriptor_diffusion",
    "descriptor_signal_change",
    "descriptor_enhancement",
    "descriptor_flow",
    "descriptor_interval_change",
    "descriptor_mass_effect",
    "descriptor_morphology",
    "descriptor_collection",
    "descriptor_necrosis",
]

PATHOLOGY_LABELS = [
    "pathology_haemorrhagic",
    "pathology_ischaemic",
    "pathology_vascular",
    "pathology_cerebrovascular",
    "pathology_treatment",
    "pathology_inflammatory_autoimmune",
    "pathology_congenital_developmental",
    "pathology_csf_disorders",
    "pathology_musculoskeletal",
    "pathology_neoplastic_paraneoplastic",
    "pathology_infectious",
    "pathology_neurodegenerative_dementia",
    "pathology_metabolic_nutritional_toxic",
    "pathology_endocrine",
    "pathology_opthalmological",
    "pathology_traumatic",
]

DOMAIN_LABELS = LOCATION_LABELS + DESCRIPTOR_LABELS + PATHOLOGY_LABELS


def minibatch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class Sectioner:
    def __init__(self, model, tokenizer, device="cpu", batch_size=32):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = (
            AutoModelForTokenClassification.from_pretrained(model)
            .eval()
            .to(self.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, device=self.device)

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        for batch in minibatch(texts, n=self.batch_size):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [
                [self.model.config.id2label[t.item()] for t in p] for p in predictions
            ]
            partitions = [self._get_partitions(t) for t in predicted_token_class]
            output = self._decode_partition(inputs, partitions)
            yield from output

    def _decode_partition(self, inputs, partitions):
        output = []
        for ids, partition in zip(inputs["input_ids"], partitions):
            default_sections = defaultdict(list)
            decoded = [
                (k, self.tokenizer.decode(ids[v], skip_special_tokens=True))
                for k, v in partition
            ]
            for k, v in decoded:
                if v:
                    default_sections[k] += [v]
            default_sections = {k: " ".join(v) for k, v in default_sections.items()}
            output.append(default_sections)
        return output

    def _get_partitions(self, token_class_list):
        indices = [(t, i) for i, t in enumerate(token_class_list) if t != "O"]
        partitions = [(l[0], slice(l[1], r[1])) for l, r in pairwise(indices)]
        if indices:
            partitions.append(
                (indices[-1][0], slice(indices[-1][1], len(token_class_list)))
            )
        return partitions


class NeuroPipeline:
    def __init__(
        self,
        ner_model,
        negation_model,
        tokenizer,
        aggregation_strategy="first",
        device="cpu",
        batch_size=32,
    ):
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_model
        ).eval()
        self.negation_model = AutoModelForTokenClassification.from_pretrained(
            negation_model
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.batch_size = batch_size
        self.ner_pipeline = pipeline(
            "ner",
            model=self.ner_model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=torch.device(device),
            batch_size=self.batch_size,
        )
        self.negation_pipeline = pipeline(
            "ner",
            model=self.negation_model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=torch.device(device),
            batch_size=self.batch_size,
        )

    def has_intersection(self, start_1, end_1, start_2, end_2):
        return (start_1 <= end_2) and (start_2 <= end_1)

    def _correct_empty_input(self, texts):
        """ner pipelines fail for empty input, add a place holder"""
        return [i if i.strip() else "<placeholder>" for i in texts]

    def _get_ner_neg_intersections(self, ner_outputs, negation_outputs):
        doc_outputs = []
        for ner_output, negation_output in zip(ner_outputs, negation_outputs):
            doc_output = []
            for ner in ner_output:
                ner["negated"] = False
                for neg in negation_output:
                    if self.has_intersection(
                        ner["start"], ner["end"], neg["start"], neg["end"]
                    ):
                        ner["negated"] = neg["entity_group"] == "negated"
                        break
                doc_output.append(ner)
            doc_outputs.append(doc_output)
        return doc_outputs

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        texts = self._correct_empty_input(texts)
        for batch in minibatch(texts, n=self.batch_size):
            ner_outputs = self.ner_pipeline(batch)
            negation_outputs = self.negation_pipeline(batch)
            batch_output = self._get_ner_neg_intersections(
                ner_outputs, negation_outputs
            )
            batch_dicts = [
                {"text": text, "ents": labels}
                for text, labels in zip(batch, batch_output)
            ]
            yield from batch_dicts


class DocConverter:
    def __init__(self, model):
        self.nlp = spacy.load(
            model, exclude=["ner", "negex", "lemmatizer", "attribute_ruler"]
        )
        Span.set_extension("negex", default=False, force=True)

    def __call__(self, pipe_output):
        """convert output from the neuro pipeline to a spacy doc"""
        doc = self.nlp(pipe_output["text"])
        ents = []
        for ent in pipe_output["ents"]:
            span = doc.char_span(ent["start"], ent["end"], label=ent["entity_group"])
            if span:
                span._.negex = ent["negated"]
                ents.append(span)
        doc.set_ents(ents)
        return doc


class RelationExtractor:
    """Pipe assigning relations between concepts and their corresponding locations.

    PATHOLOGY and DESCRIPTOR concepts tagged by an entity recogniser
    pipe are connected to their corresponding LOCATIONS by using a
    grammatical tree-traversal algorithm.

    The relations for each entity are stored in the Span.relation attribute,
    where the Span entity objects can be accessed via the Doc.ents generator.
    """

    def __init__(self, vocab, name="relex"):
        self.name = name
        self.patterns = {}
        self.matcher = DependencyMatcher(vocab)
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default=[])
        if not Span.has_extension("relation"):
            Span.set_extension("relation", getter=self.relation_getter)

    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        doc._.rel.extend(
            list(set([(token_ids[0], token_ids[-1]) for _, token_ids in matches]))
        )
        return doc

    def relation_getter(self, span):
        """given an entity span, find all relations attached to it"""
        span_rel_ids = set(
            [rels[-1] for w in span for rels in span.doc._.rel if w.i == rels[0]]
        )
        spans = []
        for e in span.doc.ents:
            for w in e:
                if w.i in span_rel_ids:
                    spans.append(e)
                    break
        return spans

    def add_patterns(self, pattern_dict):
        for name, pattern in pattern_dict.items():
            self.matcher.add(name, [pattern])

    def from_disk(self, path, exclude=tuple()):
        self.patterns = srsly.read_json(Path(path) / "patterns.json")

        deserializers_patterns = {
            "patterns": lambda p: self.add_patterns(
                srsly.read_json(p.with_suffix(".json"))
            )
        }
        from_disk(path, deserializers_patterns, {})
        return self


class DocVectorizer:
    def __init__(
        self,
        entity_labels=PATHOLOGY_LABELS + DESCRIPTOR_LABELS,
        location_labels=LOCATION_LABELS,
    ):
        self.e_mapping = dict(enumerate(entity_labels))
        self.l_mapping = dict(enumerate(["location_unk"] + location_labels))
        # assert map is just int(True/false)
        self.e_map_rev = {j: i for i, j in self.e_mapping.items()}
        self.l_map_rev = {j: i for i, j in self.l_mapping.items()}
        self.mappings = {
            "entity_map": self.e_map_rev,
            "location_map": self.l_map_rev,
            "assertion_map": {True: 1, False: 0},
        }

    def doc2array(self, doc):
        """convert a doc to a numpy count array of entity-location-assertion triplets"""
        count_array = np.zeros((len(self.e_mapping), len(self.l_mapping), 2), dtype=int)
        for e in doc.ents:
            if e.label_ in self.e_map_rev.keys():
                x = self.e_map_rev[e.label_]
                z = int(e._.negex)
                count_array[x, 0, z] += 1
                for r in e._.relation:
                    y = self.l_map_rev[r.label_]
                    count_array[x, y, z] += 1
        return count_array

    def multidoc2array(self, docs):
        return np.array([self.doc2array(i) for i in tqdm(docs)])
