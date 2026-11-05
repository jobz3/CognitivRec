.PHONY: setup lint format test test-network stage1 stage2 stage3 stage4 stages audit

# The matrix products here are small: extra BLAS threads cost CPU without saving any real time.
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1
export MKL_NUM_THREADS = 1

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-network:
	uv run pytest -m network -o addopts=""

stage1:
	uv run python -m cognitiverec.experiments.stage1_baselines

stage2:
	uv run python -m cognitiverec.experiments.stage2_cognitive

stage3:
	uv run python -m cognitiverec.experiments.stage3_trust

stage4:
	uv run python -m cognitiverec.experiments.stage4_ablation

stages: stage1 stage2 stage3 stage4

audit:
	uv run python scripts/audit_plan_code.py
