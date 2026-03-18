"""Generate train/val/test JSONL datasets."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from shell_rft.generation import GENERATORS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shell-rft datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-n", type=int, default=500)
    parser.add_argument("--val-n", type=int, default=100)
    parser.add_argument("--test-n", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", args.train_n, args.seed),
        ("val", args.val_n, args.seed + 1),
        ("test", args.test_n, args.seed + 2),
    ]

    for split_name, count, split_seed in splits:
        rng = random.Random(split_seed)
        examples = []
        generators = list(GENERATORS.values())
        per_gen = count // len(generators)
        remainder = count % len(generators)

        for i, gen_fn in enumerate(generators):
            n = per_gen + (1 if i < remainder else 0)
            examples.extend(gen_fn(n, rng))

        rng.shuffle(examples)

        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + "\n")

        print(f"Wrote {len(examples)} examples to {path}")


if __name__ == "__main__":
    main()
