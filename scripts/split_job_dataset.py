#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split job_100g_data into fixed train/test directories.

This script is meant to remove the need for passing train_ratio at runtime:
after splitting once, the codebase reads:
  - data_kunpeng/job_100g_data_train/plan_info.csv
  - data_kunpeng/job_100g_data_train/query_info.csv
  - data_kunpeng/job_100g_data_test/plan_info.csv
  - data_kunpeng/job_100g_data_test/query_info.csv
"""

import argparse
import csv
import os
import random
from typing import Set, Tuple


def _read_query_ids(query_info_path: str, delimiter: str) -> Tuple[list, str]:
    with open(query_info_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty query_info file: {query_info_path}")

        try:
            qid_idx = header.index("query_id")
        except ValueError as e:
            raise ValueError(f"query_info header missing 'query_id': {header}") from e

        query_ids = []
        for row in reader:
            if not row:
                continue
            if qid_idx >= len(row):
                continue
            qid = row[qid_idx].strip()
            if not qid:
                continue
            query_ids.append(int(float(qid)))

    return query_ids, header[qid_idx]


def _route_csv_by_query_id(
    input_path: str,
    output_path_train: str,
    output_path_test: str,
    train_query_ids: Set[int],
    delimiter: str,
    query_id_column: str = "query_id",
):
    with open(input_path, "r", encoding="utf-8", errors="replace", newline="") as fin:
        reader = csv.reader(fin, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV file: {input_path}")

        try:
            qid_idx = header.index(query_id_column)
        except ValueError as e:
            raise ValueError(f"CSV header missing '{query_id_column}' in {input_path}: {header}") from e

        os.makedirs(os.path.dirname(output_path_train), exist_ok=True)
        os.makedirs(os.path.dirname(output_path_test), exist_ok=True)

        with open(output_path_train, "w", encoding="utf-8", newline="") as ftrain, open(
            output_path_test, "w", encoding="utf-8", newline=""
        ) as ftest:
            wtrain = csv.writer(ftrain, delimiter=delimiter)
            wtest = csv.writer(ftest, delimiter=delimiter)

            wtrain.writerow(header)
            wtest.writerow(header)

            for row in reader:
                if not row:
                    continue
                if qid_idx >= len(row):
                    continue
                qid_raw = row[qid_idx].strip()
                if not qid_raw:
                    continue
                qid = int(float(qid_raw))
                if qid in train_query_ids:
                    wtrain.writerow(row)
                else:
                    wtest.writerow(row)


def split_job_dataset(
    base_dir: str,
    out_train_dir: str,
    out_test_dir: str,
    train_ratio: float,
    seed: int,
    delimiter: str = ";",
):
    query_info_path = os.path.join(base_dir, "query_info.csv")
    plan_info_path = os.path.join(base_dir, "plan_info.csv")
    if not os.path.exists(query_info_path):
        raise FileNotFoundError(query_info_path)
    if not os.path.exists(plan_info_path):
        raise FileNotFoundError(plan_info_path)

    # Read unique query IDs.
    query_ids, _ = _read_query_ids(query_info_path, delimiter=delimiter)
    unique_qids = sorted(set(query_ids))
    if not unique_qids:
        raise ValueError(f"No query_id found in {query_info_path}")

    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got: {train_ratio}")

    rng = random.Random(seed)
    rng.shuffle(unique_qids)

    split_point = int(len(unique_qids) * train_ratio)
    if split_point <= 0 or split_point >= len(unique_qids):
        raise ValueError(
            f"Split is invalid: len(unique_qids)={len(unique_qids)}, train_ratio={train_ratio}, split_point={split_point}"
        )

    train_qids = set(unique_qids[:split_point])
    test_qids = set(unique_qids[split_point:])

    print(
        f"Job dataset split: {len(train_qids)} train queries, {len(test_qids)} test queries "
        f"(train_ratio={train_ratio}, seed={seed})"
    )

    # Route query_info and plan_info rows by query_id.
    _route_csv_by_query_id(
        input_path=query_info_path,
        output_path_train=os.path.join(out_train_dir, "query_info.csv"),
        output_path_test=os.path.join(out_test_dir, "query_info.csv"),
        train_query_ids=train_qids,
        delimiter=delimiter,
        query_id_column="query_id",
    )

    # For plan_info.csv, query_id is also present as a column.
    _route_csv_by_query_id(
        input_path=plan_info_path,
        output_path_train=os.path.join(out_train_dir, "plan_info.csv"),
        output_path_test=os.path.join(out_test_dir, "plan_info.csv"),
        train_query_ids=train_qids,
        delimiter=delimiter,
        query_id_column="query_id",
    )

    # Save split summary.
    os.makedirs(os.path.dirname(out_train_dir), exist_ok=True)
    summary_path = os.path.join(os.path.dirname(out_train_dir), "job_100g_data_split_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["seed", "train_ratio", "n_train_queries", "n_test_queries"])
        writer.writerow([seed, train_ratio, len(train_qids), len(test_qids)])

    print(f"Split directories ready: {out_train_dir} and {out_test_dir}")
    print(f"Split summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Split job_100g_data into train/test directories")
    parser.add_argument("--base_dir", type=str, default="data_kunpeng/job_100g_data")
    parser.add_argument("--out_train_dir", type=str, default="data_kunpeng/job_100g_data_train")
    parser.add_argument("--out_test_dir", type=str, default="data_kunpeng/job_100g_data_test")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delimiter", type=str, default=";")
    args = parser.parse_args()

    split_job_dataset(
        base_dir=args.base_dir,
        out_train_dir=args.out_train_dir,
        out_test_dir=args.out_test_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        delimiter=args.delimiter,
    )


if __name__ == "__main__":
    main()

