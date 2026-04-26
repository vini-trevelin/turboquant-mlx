#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODEL="${MODEL:-mlx-community/Llama-3.2-3B-Instruct-4bit}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
LONGBENCH_DIR="${LONGBENCH_DIR:-assets/longbench/data}"
BENCH_MODE="${BENCH_MODE:-full}"
SEED="${SEED:-0}"

QUALITY_DATASETS="${QUALITY_DATASETS:-triviaqa_e,hotpotqa_e,2wikimqa_e}"
CALIBRATION_DATASETS="${CALIBRATION_DATASETS:-multifieldqa_en,multi_news}"
REQUIRED_DATASETS="${REQUIRED_DATASETS:-triviaqa_e hotpotqa_e 2wikimqa_e multifieldqa_en multi_news}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip
"$VENV_DIR/bin/python" -m pip install -e ".[dev]" "huggingface_hub>=0.20"

mkdir -p "$LONGBENCH_DIR"

"$VENV_DIR/bin/python" - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import os

from huggingface_hub import hf_hub_download

out = Path(os.environ.get("LONGBENCH_DIR", "assets/longbench/data"))
required = os.environ.get(
    "REQUIRED_DATASETS",
    "triviaqa_e hotpotqa_e 2wikimqa_e multifieldqa_en multi_news",
).split()

missing = [name for name in required if not (out / f"{name}.jsonl").exists()]
if not missing:
    print(f"LongBench files already present in {out}")
    raise SystemExit(0)

zip_path = hf_hub_download(
    repo_id="THUDM/LongBench",
    filename="data.zip",
    repo_type="dataset",
)
out.mkdir(parents=True, exist_ok=True)
with ZipFile(zip_path) as archive:
    names = set(archive.namelist())
    for dataset in required:
        source = f"data/{dataset}.jsonl"
        if source not in names:
            raise FileNotFoundError(f"{source} not found in {zip_path}")
        target = out / f"{dataset}.jsonl"
        with archive.open(source) as src, target.open("wb") as dst:
            dst.write(src.read())
        print(target)
PY

if [[ "$BENCH_MODE" == "smoke" ]]; then
  exec "$VENV_DIR/bin/turboquant-report" \
    --model "$MODEL" \
    --longbench-source-dir "$LONGBENCH_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --context-tiers 512 \
    --quality-datasets "$QUALITY_DATASETS" \
    --calibration-datasets "$CALIBRATION_DATASETS" \
    --quality-examples-per-dataset 1 \
    --calibration-examples-per-dataset 1 \
    --parity-examples-per-dataset 1 \
    --needle-seeds-per-position 1 \
    --headline-only \
    --seed "$SEED"
fi

if [[ "$BENCH_MODE" != "full" ]]; then
  echo "BENCH_MODE must be 'full' or 'smoke', got: $BENCH_MODE" >&2
  exit 2
fi

exec "$VENV_DIR/bin/turboquant-report" \
  --model "$MODEL" \
  --longbench-source-dir "$LONGBENCH_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --context-tiers "${CONTEXT_TIERS:-512,2048,4096}" \
  --quality-datasets "$QUALITY_DATASETS" \
  --calibration-datasets "$CALIBRATION_DATASETS" \
  --quality-examples-per-dataset "${QUALITY_EXAMPLES_PER_DATASET:-10}" \
  --calibration-examples-per-dataset "${CALIBRATION_EXAMPLES_PER_DATASET:-2}" \
  --parity-examples-per-dataset "${PARITY_EXAMPLES_PER_DATASET:-1}" \
  --quality-max-tokens "${QUALITY_MAX_TOKENS:-24}" \
  --needle-max-tokens "${NEEDLE_MAX_TOKENS:-12}" \
  --needle-seeds-per-position "${NEEDLE_SEEDS_PER_POSITION:-4}" \
  --seed "$SEED"
