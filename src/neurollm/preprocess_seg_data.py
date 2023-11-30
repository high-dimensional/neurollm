#!/usr/bin/env python
"""Script title.

This text describes the purpose of the script
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_input(location):
    """load in data"""
    columns = [
        "Narrative",
        "report_header",
        "report_indications",
        "report_metareport",
        "report_body",
        "report_tail",
    ]
    df = pd.read_csv(location, low_memory=False, usecols=columns)
    return df


def transform_data(data, size):
    """perform the necessary transformation on the input data"""
    sample = data.sample(size).fillna("")
    return sample


def output_results(data, outdir):
    """output analysis, save to file or send to stdout"""
    train, dev = train_test_split(data, test_size=0.2, random_state=42)
    dev, test = train_test_split(dev, test_size=0.5, random_state=42)
    train.to_csv(outdir / "train.csv", index=False)
    dev.to_csv(outdir / "dev.csv", index=False)
    test.to_csv(outdir / "test.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="report CSV", type=Path)
    parser.add_argument("-s", "--size", help="dataset size", type=int, default=1000)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    args = parser.parse_args()
    if not args.outdir.exists():
        args.outdir.mkdir()
    data = load_input(args.input)
    transformed_data = transform_data(data, args.size)
    output_results(transformed_data, args.outdir)


if __name__ == "__main__":
    main()
