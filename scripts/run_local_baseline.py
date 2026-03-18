"""Run prompt-only baseline evaluation on held-out data.

Sends each test example to the Fireworks API, evaluates the model's response
using the evaluator pipeline, and reports aggregate metrics.

Requires FIREWORKS_API_KEY environment variable.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add evaluators/ to sys.path so we can import shell_rft_v0 as a package.
_EVAL_DIR = str(Path(__file__).resolve().parent.parent / "evaluators")
sys.path.insert(0, _EVAL_DIR)

from shell_rft_v0.normalize import normalize_stdout  # noqa: E402
from shell_rft_v0.policy import extract_single_command, validate_command  # noqa: E402
from shell_rft_v0.sandbox import materialize_workspace, run_command  # noqa: E402

import shutil  # noqa: E402
import tempfile  # noqa: E402

from fireworks.client import Fireworks  # noqa: E402


def load_examples(path: str) -> list[dict]:
    """Load JSONL examples from a file."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def call_model(
    client: Fireworks,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """Call the Fireworks API and return the assistant's response text.

    Strips <think>...</think> reasoning blocks that Qwen3 emits by default.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    # Strip Qwen3 reasoning blocks.
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text


def evaluate_response(
    response_text: str,
    workspace_spec: dict,
    expected_stdout: str,
    task_type: str,
) -> dict:
    """Evaluate a model response through the evaluator pipeline.

    Returns a dict with stage-by-stage results:
      - command: extracted command or None
      - parse_ok: bool
      - policy_ok: bool (None if parse failed)
      - policy_reason: str or None
      - exec_ok: bool (None if policy failed)
      - exit_code: int or None
      - actual_stdout: str or None
      - match: bool (None if exec failed)
      - score: float (1.0 or 0.0)
    """
    result = {
        "response_text": response_text,
        "command": None,
        "parse_ok": False,
        "policy_ok": None,
        "policy_reason": None,
        "exec_ok": None,
        "exit_code": None,
        "actual_stdout": None,
        "match": None,
        "score": 0.0,
    }

    # Stage 1: Extract command.
    command = extract_single_command(response_text)
    if command is None:
        return result
    result["command"] = command
    result["parse_ok"] = True

    # Stage 2: Validate policy.
    violation = validate_command(command)
    if violation is not None:
        result["policy_ok"] = False
        result["policy_reason"] = violation
        return result
    result["policy_ok"] = True

    # Stage 3: Execute in sandbox.
    workdir = tempfile.mkdtemp(prefix="shell_rft_baseline_")
    try:
        materialize_workspace(workspace_spec, workdir)
        exit_code, stdout, stderr = run_command(command, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    result["exit_code"] = exit_code
    if exit_code != 0:
        result["exec_ok"] = False
        return result
    result["exec_ok"] = True
    result["actual_stdout"] = stdout

    # Stage 4: Compare output.
    actual = normalize_stdout(stdout, task_type=task_type)
    expected = normalize_stdout(expected_stdout, task_type=task_type)
    result["match"] = actual == expected
    result["score"] = 1.0 if result["match"] else 0.0

    return result


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-example results."""
    n = len(results)
    if n == 0:
        return {}

    metrics = {
        "total": n,
        "exact_match": sum(1 for r in results if r["score"] == 1.0),
        "parse_success": sum(1 for r in results if r["parse_ok"]),
        "policy_pass": sum(1 for r in results if r["policy_ok"] is True),
        "policy_fail": sum(1 for r in results if r["policy_ok"] is False),
        "exec_success": sum(1 for r in results if r["exec_ok"] is True),
        "exec_fail": sum(1 for r in results if r["exec_ok"] is False),
        "output_match": sum(1 for r in results if r["match"] is True),
        "output_mismatch": sum(1 for r in results if r["match"] is False),
    }

    metrics["exact_match_rate"] = metrics["exact_match"] / n
    metrics["parse_success_rate"] = metrics["parse_success"] / n
    metrics["mean_reward"] = sum(r["score"] for r in results) / n

    # Per-task-family breakdown.
    by_family: dict[str, list[dict]] = {}
    for r in results:
        family = r.get("task_type", "unknown")
        by_family.setdefault(family, []).append(r)

    family_metrics = {}
    for family, family_results in sorted(by_family.items()):
        fn = len(family_results)
        em = sum(1 for r in family_results if r["score"] == 1.0)
        family_metrics[family] = {
            "total": fn,
            "exact_match": em,
            "exact_match_rate": em / fn,
        }
    metrics["by_family"] = family_metrics

    return metrics


def print_summary(metrics: dict) -> None:
    """Print a human-readable summary table."""
    n = metrics["total"]
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"  Total examples:      {n}")
    print(f"  Exact match:         {metrics['exact_match']}/{n} ({metrics['exact_match_rate']:.1%})")
    print(f"  Mean reward:         {metrics['mean_reward']:.3f}")
    print(f"  Parse success:       {metrics['parse_success']}/{n} ({metrics['parse_success_rate']:.1%})")
    print(f"  Policy pass:         {metrics['policy_pass']}/{n}")
    print(f"  Policy fail:         {metrics['policy_fail']}/{n}")
    print(f"  Exec success:        {metrics['exec_success']}/{n}")
    print(f"  Exec fail:           {metrics['exec_fail']}/{n}")
    print(f"  Output match:        {metrics['output_match']}/{n}")
    print(f"  Output mismatch:     {metrics['output_mismatch']}/{n}")
    print()
    print("Per task family:")
    print(f"  {'Family':<20} {'Match':>6} {'Total':>6} {'Rate':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8}")
    for family, fm in metrics["by_family"].items():
        print(f"  {family:<20} {fm['exact_match']:>6} {fm['total']:>6} {fm['exact_match_rate']:>8.1%}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prompt-only baseline evaluation on held-out data."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/qwen3-8b",
        help="Fireworks model identifier",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/test.jsonl",
        help="Path to JSONL dataset to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/baseline_results.json",
        help="Path to save detailed results JSON",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for model response (needs headroom for Qwen3 reasoning)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of examples to evaluate (0 = all)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY environment variable is required.")
        sys.exit(1)

    # Load examples.
    examples = load_examples(args.dataset)
    if args.limit > 0:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples from {args.dataset}")

    # Initialize Fireworks client.
    client = Fireworks(api_key=api_key)

    # Evaluate each example.
    results = []
    for i, ex in enumerate(examples):
        task_type = ex.get("task_type", "unknown")
        messages = ex["messages"]

        print(f"  [{i+1}/{len(examples)}] {task_type}...", end=" ", flush=True)

        try:
            response_text = call_model(
                client,
                args.model,
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"API error: {e}")
            result = {
                "response_text": "",
                "command": None,
                "parse_ok": False,
                "policy_ok": None,
                "policy_reason": None,
                "exec_ok": None,
                "exit_code": None,
                "actual_stdout": None,
                "match": None,
                "score": 0.0,
                "error": str(e),
            }
            result["task_type"] = task_type
            results.append(result)
            continue

        result = evaluate_response(
            response_text,
            ex["workspace_spec"],
            ex["expected_stdout"],
            task_type,
        )
        result["task_type"] = task_type

        status = "MATCH" if result["score"] == 1.0 else "MISS"
        print(f"{status} (cmd: {result['command']})")

        results.append(result)

    # Compute and display metrics.
    metrics = compute_metrics(results)
    print_summary(metrics)

    # Save detailed results.
    output_data = {
        "model": args.model,
        "dataset": args.dataset,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "metrics": metrics,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
