#!/usr/bin/env python
"""Train a ner model.

Train the huggingface NER model on a set of labelled NER data
"""
import argparse
from pathlib import Path

from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


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
    args = parser.parse_args()
    ds = (
        Dataset.load_from_disk(args.data_dir)
        .class_encode_column("normality_class")
        .rename_column("normality_class", "label")
        .rename_column("Narrative", "text")
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_data = ds.map(preprocess_function, batched=True)
    collator = DataCollatorWithPadding(tokenizer)
    id2label = {
        tokenized_data.features["label"].str2int(i): i
        for i in tokenized_data.features["label"].names
    }
    label2id = {j: i for i, j in id2label.items()}
    split_data = tokenized_data.train_test_split(test_size=0.2, shuffle=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
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
        train_dataset=split_data["train"],
        eval_dataset=split_data["test"],
    )

    trainer.train()


if __name__ == "__main__":
    main()
