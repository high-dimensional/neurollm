#!/usr/bin/env python
"""Train a segmentation model.

This text describes the purpose of the script
"""
import argparse
from pathlib import Path

from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)


def load_input(location):
    """load in data or take input from stdin"""
    ds_train = Dataset.from_csv(str(location / "train.csv"), keep_in_memory=True)
    ds_eval = Dataset.from_csv(str(location / "dev.csv"), keep_in_memory=True)
    return ds_train, ds_eval


def transform_data(data, model):
    """perform the necessary transformation on the input data"""
    tokenizer = AutoTokenizer.from_pretrained(model)

    def tokenize_and_label(datum):
        labels = []  # [-100]
        input_ids = []  # [tokenizer.bos_token_id]
        for i, section in enumerate(
            [
                "report_header",
                "report_indications",
                "report_metareport",
                "report_body",
                "report_tail",
            ]
        ):
            text = datum[section] if datum[section] else " "
            token_dict = tokenizer(text, add_special_tokens=False, truncation=True)
            if len(token_dict["input_ids"]) > 0:
                input_ids.extend(token_dict["input_ids"])
                section_labels = [0] * len(token_dict["input_ids"])
                section_labels[0] = i + 1
                labels.extend(section_labels)
        # labels.append(-100)
        # input_ids.append(tokenizer.eos_token_id)
        input_ids = input_ids[: tokenizer.model_max_length - 1]
        labels = labels[: tokenizer.model_max_length - 1]
        return {"input_ids": input_ids, "labels": labels}

    preprocessed_data = data.map(
        tokenize_and_label,
        batched=False,
        remove_columns=[
            "report_header",
            "report_indications",
            "report_metareport",
            "report_body",
            "report_tail",
        ],
    )

    return preprocessed_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="report CSV", type=Path)
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
    if not args.outdir.exists():
        args.outdir.mkdir()
    train_data, eval_data = load_input(args.data_dir)
    train_dataset = transform_data(train_data, args.model)
    print(train_dataset[0])
    eval_dataset = transform_data(eval_data, args.model)
    base_labels = [
        "O",
        "report_header",
        "report_indications",
        "report_metareport",
        "report_body",
        "report_tail",
    ]
    id2label = dict(enumerate(base_labels))
    label2id = {v: k for k, v in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
