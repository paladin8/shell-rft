#!/bin/bash
# Evaluate an RFT-tuned model against the test set and compare with baseline.
#
# Usage:
#   ./scripts/evaluate_rft_model.sh <model-or-deployment-id> [output-file]
#
# Examples:
#   # Using a deployment (required for LoRA fine-tuned models):
#   ./scripts/evaluate_rft_model.sh accounts/jeffreyw314159-da5a28/deployments/bzo15dj5
#
#   # With custom output path:
#   ./scripts/evaluate_rft_model.sh accounts/jeffreyw314159-da5a28/deployments/bzo15dj5 data/rft_run2_results.json
#
# Notes:
#   - LoRA fine-tuned models must be deployed first (on-demand deployment)
#   - Use the deployment ID, not the model name, as the model parameter
#   - Requires FIREWORKS_API_KEY in .env or environment
#   - max-tokens=4096 gives room for Qwen3's <think> reasoning blocks

set -euo pipefail

MODEL="${1:?Usage: $0 <model-or-deployment-id> [output-file]}"
OUTPUT="${2:-data/rft_results.json}"

echo "Evaluating model: $MODEL"
echo "Output: $OUTPUT"
echo

uv run python scripts/run_local_baseline.py \
  --model "$MODEL" \
  --dataset data/test.jsonl \
  --output "$OUTPUT" \
  --max-tokens 4096

echo
echo "=== Comparison with baseline ==="
uv run python scripts/compare_results.py data/baseline_results.json "$OUTPUT"
