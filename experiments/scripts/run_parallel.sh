#!/usr/bin/env bash
# Run prepare, run, and aggregate for every dataset and seed in parallel on CPU.
# Usage: experiments/scripts/run_parallel.sh [ROOT] [PREPARED] [RUNS] [AGGREGATE]
# SOCP_WORKERS sets the number of parallel jobs (default 40). Each job uses one
# BLAS thread. The dataset and seed lists match experiments/config.yaml.
set -euo pipefail

SOCP_ROOT=${1:-$PWD}
SOCP_PREPARED=${2:-data/prepared}
SOCP_RUNS=${3:-results/runs}
SOCP_AGGREGATE=${4:-results/aggregate}
SOCP_WORKERS=${SOCP_WORKERS:-40}
SOCP_PYTHON=${SOCP_PYTHON:-$SOCP_ROOT/.venv/bin/python}
SOCP_LOGS=$SOCP_ROOT/results/logs
SOCP_STATUS=$SOCP_ROOT/results/status.txt
export SOCP_PYTHON SOCP_LOGS SOCP_PREPARED SOCP_RUNS

cd "$SOCP_ROOT"
mkdir -p "$SOCP_LOGS" "$SOCP_PREPARED" "$SOCP_RUNS" "$SOCP_AGGREGATE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export MPLCONFIGDIR=$SOCP_ROOT/.runtime/matplotlib
export PYTHONPATH=$SOCP_ROOT
mkdir -p "$MPLCONFIGDIR"

SOCP_DATASETS=(
  bike bio california_housing concrete mpg
  cifar10 mnist covertype cifar100 imagenet1k
)
SOCP_SEEDS=(42 123 288 327 456 555 690 761 832 999)

printf 'preparing started=%s workers=%s\n' "$(date -Is)" "$SOCP_WORKERS" >"$SOCP_STATUS"
for SOCP_DATASET in "${SOCP_DATASETS[@]}"; do
  for SOCP_SEED in "${SOCP_SEEDS[@]}"; do
    printf '%s %s\n' "$SOCP_DATASET" "$SOCP_SEED"
  done
done | xargs -P "$SOCP_WORKERS" -n 2 bash -c '
  dataset=$1
  seed=$2
  log=$SOCP_LOGS/prepare_${dataset}_${seed}.log
  "$SOCP_PYTHON" -m experiments.cli prepare \
    --config experiments/config.yaml --profile full \
    --datasets "$dataset" --seeds "$seed" --output "$SOCP_PREPARED" \
    >"$log" 2>&1
' _

printf 'running started=%s workers=%s\n' "$(date -Is)" "$SOCP_WORKERS" >"$SOCP_STATUS"
for SOCP_DATASET in "${SOCP_DATASETS[@]}"; do
  for SOCP_SEED in "${SOCP_SEEDS[@]}"; do
    printf '%s %s\n' "$SOCP_DATASET" "$SOCP_SEED"
  done
done | xargs -P "$SOCP_WORKERS" -n 2 bash -c '
  dataset=$1
  seed=$2
  log=$SOCP_LOGS/run_${dataset}_${seed}.log
  "$SOCP_PYTHON" -m experiments.cli run \
    --config experiments/config.yaml --profile full \
    --datasets "$dataset" --seeds "$seed" \
    --prepared "$SOCP_PREPARED" --output "$SOCP_RUNS" \
    >"$log" 2>&1
' _

printf 'aggregating started=%s\n' "$(date -Is)" >"$SOCP_STATUS"
"$SOCP_PYTHON" -m experiments.cli aggregate \
  --config experiments/config.yaml --profile full \
  --runs "$SOCP_RUNS" --output "$SOCP_AGGREGATE" \
  >"$SOCP_LOGS/aggregate.log" 2>&1

printf 'complete finished=%s\n' "$(date -Is)" >"$SOCP_STATUS"
