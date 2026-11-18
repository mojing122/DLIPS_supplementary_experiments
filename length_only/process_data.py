"""Filter the released QA token counts to one eligible pair per prompt."""

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("source_row", "group_id", "prompt_tokens", "faithful_tokens", "hallucinated_tokens")
MIN_ANSWER_TOKENS = 5
MAX_LENGTH_GAP = 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "qa_token_counts.csv")
    parser.add_argument("--output", type=Path, default=Path("qa_filtered_pairs.csv"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")

    selected = []
    seen_groups = set()
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source, group, prompt, faithful, hallucinated = (int(row[key]) for key in FIELDS)
            if min(faithful, hallucinated) < MIN_ANSWER_TOKENS or abs(faithful - hallucinated) > MAX_LENGTH_GAP:
                continue
            if group not in seen_groups:
                selected.append((source, group, prompt, faithful, hallucinated))
                seen_groups.add(group)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(selected)
    print(f"Saved {len(selected)} eligible pairs to {args.output}")


if __name__ == "__main__":
    main()
