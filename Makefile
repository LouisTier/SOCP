CONFIG := experiments/config.yaml
PREPARED ?= data/prepared
RUNS ?= results/runs
AGGREGATE ?= results/aggregate
DATASET ?= all
DATASETS ?=
SEEDS ?=
PYTHON_SOURCES := socp experiments

.PHONY: install download prepare run aggregate smoke check format

install:
	uv sync

download:
	uv run python -m experiments.download_data --dataset $(DATASET)

prepare:
	uv run socp-experiments prepare --config $(CONFIG) --profile full --output $(PREPARED) $(if $(DATASETS),--datasets $(DATASETS)) $(if $(SEEDS),--seeds $(SEEDS))

run:
	uv run socp-experiments run --config $(CONFIG) --profile full --prepared $(PREPARED) --output $(RUNS) $(if $(DATASETS),--datasets $(DATASETS)) $(if $(SEEDS),--seeds $(SEEDS))

aggregate:
	uv run socp-experiments aggregate --config $(CONFIG) --profile full --runs $(RUNS) --output $(AGGREGATE) $(if $(DATASETS),--datasets $(DATASETS)) $(if $(SEEDS),--seeds $(SEEDS))

smoke:
	uv run socp-experiments prepare --profile smoke --output /tmp/socp-smoke-prepared
	uv run socp-experiments run --profile smoke --prepared /tmp/socp-smoke-prepared --output /tmp/socp-smoke-runs
	uv run socp-experiments aggregate --profile smoke --runs /tmp/socp-smoke-runs --output /tmp/socp-smoke-aggregate

check:
	uv run ruff format --check $(PYTHON_SOURCES)
	uv run ruff check $(PYTHON_SOURCES)
	uv run python -m mypy socp experiments

format:
	uv run ruff format $(PYTHON_SOURCES)
	uv run ruff check --fix $(PYTHON_SOURCES)
