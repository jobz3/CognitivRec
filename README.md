# CognitiveRec

Brain-inspired recommendation: can memory, attention, reward prediction and exploration keep recommendation quality competitive while improving fairness, explainability and reliability?

Group E project for Human Centred Applications and Technologies, Otto von Guericke University Magdeburg.

| Model | Description |
|---|---|
| A | Popularity |
| B1, B2 | ItemKNN and SVD matrix factorisation |
| C | Cognitive: long-term and short-term memory, attention, RPE-gated plasticity |
| D | Model C plus a UCB exploration bonus |

## Quick start

**Local**, with [uv](https://docs.astral.sh/uv/):

```bash
make setup     # create the environment
make test      # offline tests
make stage1    # or stage2, stage3, stage4, or `make stages` for all four
```

Without uv, run `pip install -e . pytest ruff nbformat`, then the commands inside the `Makefile`.

On Windows PowerShell, run `uv sync --locked` and `uv run python -m cognitiverec.experiments`.
Add `--quick` for a local smoke run or `--split per_user` to repeat tuning, ranking, fairness and
logged replay under the legacy split.
These variants write separate tuning files under `results/`.

Run `uv run python -m cognitiverec.experiments --followup` for the rolling time-cutoff study with
EASE and the matched history profiles. Add `--quick` to check the workflow on one seed and a small
search grid. Full and smoke runs use separate directories.

Pass the completed rolling run directory to `--exploration-study` with the same CLI. The study
holds both scorers fixed and selects their exploration policies on validation. The notebook runs
core stages, rolling windows, the exploration study, figure export and the explanation demo in
order.

**Notebook.** Open `notebooks/CognitiveRec_GroupE.ipynb` with the project's local Python
environment. For Colab, prepare the approved repository copy under `/content/CognitivRec` and
follow the setup cell's instructions for a CPU runtime. The notebook uses a session-specific
results directory and calls the same modules as the CLI.

MovieLens 100K downloads to `data/` on first use. Each run writes to `results/<stage>/<timestamp>/`, including the exact configuration and git commit. Git ignores both folders. A fresh clone can regenerate the results and publication figures with the commands in the project report.

## Stages

| Command | What it does | Owners | Runtime* |
|---|---|---|---|
| `make stage1` | Tune baselines on validation, report on test | Jobin, Mudit | about 45 s |
| `make stage2` | Tune Models C and D on validation | Mudit, Abhinav | about 2 min |
| `make stage3` | RQ1 to RQ4 for every model on test | All | about 2.5 min |
| `make stage4` | Ablation study and trade-off curves | Mudit, Abhinav | about 5 min |

\* Measured on one CPU core of a desktop machine, about 10 minutes in total. Colab may be slower.

Stages 1 and 2 write the selected hyperparameters to `configs/tuned/`, which is committed, and stages 3 and 4 read them. So you can run stage 3 straight after cloning.

The rolling study tunes again inside each window and saves its selections alongside the metrics.
Its UTC run ID identifies the execution, while `windows.csv` records the UTC boundaries of the
input rating windows. The run also records per-seed metrics, paired comparisons, example cases,
dependency versions, and data and source checksums.

The exploration study records its exact rolling source run, validation grid, selected weights,
four test cells, per-user metrics and paired accuracy effects. Use
`uv run python -m cognitiverec.experiments.paper_figures --help` to re-export the SVG and 300-dpi
PNG figures from completed runs. The project report lists the canonical figure bundle and
provides captions.

## Who owns what

| Area | Files | Owner |
|---|---|---|
| Data and split | `src/cognitiverec/data/` | Mudit Khandelwal |
| Models A, B1, B2 | `models/popularity.py`, `item_knn.py`, `svd.py` | Jobin Roy |
| Model C | `models/cognitive.py` | Mudit Khandelwal |
| Model D exploration | `models/exploration.py` | Sai Abhinav Kanagala |
| RQ1 quality, protocol, statistics | `evaluation/quality.py`, `protocol.py`, `stats.py` | Shared |
| RQ2 fairness | `evaluation/fairness.py` | Jobin Roy |
| RQ3 explainability | `evaluation/explainability.py` | Sai Abhinav Kanagala |
| RQ4 reliability | `evaluation/reliability.py` | Mudit Khandelwal |
| Report figures | `plots.py` | Sai Abhinav Kanagala |

## Adding or changing a model

Implement the `Recommender` interface in `models/base.py` and register the model in
`models/__init__.py`. Add a model specification to the experiment that should use it.
For RQ3, also implement `attributions`, `evidence_items`, and `score_without`.

Model fitting and updates live in `models/`. Timestamp cutoffs live in `data/split.py`. Candidate
masking and metrics live in `evaluation/`. Experiment entry points tune models, evaluate windows
and write artifacts. Study windows and search budgets are specified in `configs/`.

Models never mask candidates. The harness does that, so all models are compared on the same candidate items.

## Working together

- Branch for each change, and run `make lint test` before opening a pull request. CI runs the same checks.
- Tune on validation only. The tests exercise tuning with the test frame absent.
- Keep experiment logic in `src/` and notebook cells focused on running and presenting it.
- `make test-network` runs extra checks against the real dataset, including that baselines still reproduce their known numbers.

## Documentation

- [Project report](PROJECT_REPORT.md) covers methodology, results, figure captions, workstream
  ownership and limitations.
