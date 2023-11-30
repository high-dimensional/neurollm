#!/usr/bin/env python
"""Train a negation model.

Train the huggingface NER model on a set of labelled negation data.
This model trains negation only on a certain subset of entity classes.
"""
import argparse
from pathlib import Path

from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)


class TokenizeAligner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples[f"numeric_ner_tags"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="report data dir", type=Path)
    parser.add_argument("model", help="base model", type=str)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    parser.add_argument("--batch", help="batch size", type=int, default=8)
    parser.add_argument("--epochs", help="epochs", type=int, default=1)
    parser.add_argument("--lr", help="learn rate", type=float, default=0.0001)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.00001)
    parser.add_argument("--warmup", help="warmup steps", type=int, default=100)
    parser.add_argument("--dropout", help="classifier dropout", type=float, default=0.2)
    args = parser.parse_args()
    ds = Dataset.load_from_disk(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, add_prefix_space=False
    )
    id2label = {i: j for i, j in enumerate(set([b for a in ds["ner_tags"] for b in a]))}
    label2id = {v: k for k, v in id2label.items()}
    ds = ds.map(
        lambda x: {"numeric_ner_tags": [label2id[i] for i in x["ner_tags"]]},
        batched=False,
        num_proc=16,
    )
    tokenize_and_align = TokenizeAligner(tokenizer)
    ds_aligned = ds.map(tokenize_and_align, batched=True)
    ds_split = ds_aligned.train_test_split(test_size=0.2, shuffle=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=args.dropout,
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    train_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        warmup_steps=args.warmup,
        weight_decay=args.wd,
        logging_dir=args.outdir / "logs",
        logging_steps=100,
        eval_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
    )
    if not args.outdir.exists():
        args.outdir.mkdir()
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=ds_split["train"],
        eval_dataset=ds_split["test"],
    )

    trainer.train()


if __name__ == "__main__":
    main()
