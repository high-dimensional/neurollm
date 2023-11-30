#!/usr/bin/env python
"""Preprocess normality classifier data.

prepare reports for llm based multiclass classifier
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset


def load_input(location):
    """load in data"""
    columns = [
        "Narrative",
        "report_header",
        "report_indications",
        "report_metareport",
        "report_body",
        "report_tail",
        "IS_MISSING",
        "IS_COMPARATIVE",
        "IS_NORMAL",
        "IS_NORMAL_FOR_AGE",
        "_X",
        "_Y",
    ]
    df = pd.read_csv(location, low_memory=False, usecols=columns)
    return df


def transform_data(data, size):
    """perform the necessary transformation on the input data"""
    data = data.assign(normality_class="ABNORMAL")
    data.loc[data["IS_MISSING"] == True, "normality_class"] = "IS_MISSING"
    data.loc[data["IS_COMPARATIVE"] == True, "normality_class"] = "IS_COMPARATIVE"
    data.loc[data["IS_NORMAL"] == True, "normality_class"] = "IS_NORMAL"
    data.loc[data["IS_NORMAL_FOR_AGE"] == True, "normality_class"] = "IS_NORMAL_FOR_AGE"
    n_class_samples = size // 5
    sample = data.groupby("normality_class").sample(n_class_samples)
    ds = Dataset.from_pandas(sample[["Narrative", "normality_class"]])
    return ds


def output_results(data, outdir):
    """output analysis, save to file or send to stdout"""
    ds = Dataset.from_pandas(pd.DataFrame(data=data))
    ds.save_to_disk(outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="report CSV", type=Path)
    parser.add_argument("-s", "--size", help="dataset size", type=int, default=1000)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    args = parser.parse_args()
    data = load_input(args.input)
    transformed_data = transform_data(data, args.size)
    if not args.outdir.exists():
        args.outdir.mkdir()
    output_results(transformed_data, args.outdir)


if __name__ == "__main__":
    main()
