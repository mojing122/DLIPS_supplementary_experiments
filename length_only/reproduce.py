"""Train and evaluate the length-only detector on prefiltered QA pairs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIELDS = ("source_row", "group_id", "prompt_tokens", "faithful_tokens", "hallucinated_tokens")
DEFAULT_SEED = 246
THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(244, 249), default=DEFAULT_SEED)
    parser.add_argument("--data", type=Path, default=Path("qa_filtered_pairs.csv"))
    args = parser.parse_args()
    seed = args.seed
    if not args.data.is_file():
        parser.error(f"{args.data} not found. Generate it with process_data.py first.")
    with args.data.open(newline="", encoding="utf-8") as handle:
        rows = [tuple(int(row[key]) for key in FIELDS) for row in csv.DictReader(handle)]

    if len(rows) < 400:
        parser.error(f"{args.data} must contain at least 400 filtered pairs.")
    records = np.asarray(rows)
    source_rows, groups, prompt = records[:, :3].T
    if len(set(groups)) != len(rows) or len(set(source_rows)) != len(rows):
        parser.error(f"{args.data} must contain one pair per prompt group and unique source rows.")
    pair_tokens = records[:, 3:]
    chosen = np.random.default_rng(seed).permutation(len(rows))[:400]
    splits = np.split(chosen, [240, 320])
    train, _, test = [np.column_stack((2 * p, 2 * p + 1)).ravel() for p in splits]

    prompt_tokens = prompt.repeat(2)
    answer_tokens = pair_tokens.ravel()
    x = np.column_stack((prompt_tokens, answer_tokens, prompt_tokens + answer_tokens))
    y = np.tile([0, 1], len(rows))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    model.fit(x[train], y[train])
    scores = model.predict_proba(x[test])[:, 1]
    result = dict(seed=seed, threshold=THRESHOLD, eligible_pairs=len(rows), sampled_pairs=len(chosen),
                  f1=float(f1_score(y[test], scores >= THRESHOLD)), auroc=float(roc_auc_score(y[test], scores)))

    out = Path("results")
    out.mkdir(exist_ok=True)
    with (out / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("partition", *FIELDS))
        for name, split in zip(("train", "validation", "test"), splits):
            writer.writerows((name, *rows[i]) for i in split)
    with (out / "test_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source_row", "label", "prompt_tokens", "answer_tokens", "total_tokens", "score"))
        writer.writerows((source_rows[i // 2], y[i], *x[i], score) for i, score in zip(test, scores))
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"F1: {result['f1']:.6f}")
    print(f"AUROC: {result['auroc']:.6f}")


if __name__ == "__main__":
    main()
