"""Compare baseline and RFT results side by side."""

import json
import sys


def main():
    baseline_path = sys.argv[1] if len(sys.argv) > 1 else "data/baseline_results.json"
    rft_path = sys.argv[2] if len(sys.argv) > 2 else "data/rft_results.json"

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(rft_path) as f:
        rft = json.load(f)

    bm = baseline["metrics"]
    rm = rft["metrics"]

    print("=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print(f"  Baseline model: {baseline['model']}")
    print(f"  RFT model:      {rft['model']}")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>10} {'RFT':>10} {'Delta':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for key in ["exact_match_rate", "mean_reward", "parse_success_rate"]:
        bv = bm[key]
        rv = rm[key]
        delta = rv - bv
        sign = "+" if delta > 0 else ""
        print(f"{key:<25} {bv:>10.1%} {rv:>10.1%} {sign}{delta:>9.1%}")

    print()
    print("Per task family:")
    print(f"{'Family':<20} {'Baseline':>10} {'RFT':>10} {'Delta':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for family in sorted(bm["by_family"].keys()):
        bv = bm["by_family"][family]["exact_match_rate"]
        rv = rm["by_family"].get(family, {}).get("exact_match_rate", 0.0)
        delta = rv - bv
        sign = "+" if delta > 0 else ""
        print(f"{family:<20} {bv:>10.1%} {rv:>10.1%} {sign}{delta:>9.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
