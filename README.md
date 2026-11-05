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

**Google Colab:** open `notebooks/CognitiveRec_GroupE.ipynb` from GitHub and run the cells on a CPU runtime. It clones this repo and needs nothing extra installed.

MovieLens 100K downloads to `data/` on first use. Each run writes to `results/<stage>/<timestamp>/`, including the exact configuration and git commit. Neither folder is committed.

## Stages

| Command | What it does | Owners | Runtime* |
|---|---|---|---|
| `make stage1` | Tune baselines on validation, report on test | Jobin, Mudit | about 45 s |
| `make stage2` | Tune Models C and D on validation | Mudit, Abhinav | about 2 min |
| `make stage3` | RQ1 to RQ4 for every model on test | All | about 2.5 min |
| `make stage4` | Ablation study and trade-off curves | Mudit, Abhinav | about 5 min |

\* Measured on one CPU core of a desktop machine, about 10 minutes in total. Colab may be slower.

Stages 1 and 2 write the selected hyperparameters to `configs/tuned/`, which is committed, and stages 3 and 4 read them. So you can run stage 3 straight after cloning.

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

Implement the `Recommender` interface in `models/base.py`: `fit`, `score`, and optionally `observe` and `record_impression`. Add `attributions` and `score_without` if the model can be tested on RQ3, register it in `models/__init__.py`, and it goes through every evaluation automatically.

Models never mask candidates. The harness does that, so all models are compared on the same candidate items.

## Working together

- Branch for each change, and run `make lint test` before opening a pull request. CI runs the same checks.
- Tune on validation only. `make test` fails if a tuning function reads the test split.
- Change code in `src/`, not in the notebook, and don't commit notebook outputs. A test checks this too.
- `make test-network` runs extra checks against the real dataset, including that baselines still reproduce their known numbers.

## Documentation

- [`docs/methodology.md`](docs/methodology.md): evaluation protocol, fairness definition, hypotheses and limitations.
- [`docs/changes-from-plan.md`](docs/changes-from-plan.md): how this code differs from the original plan's Colab cells and why, plus open questions for each owner. `make audit` reproduces its numbers.

## Timeline

| Date | Milestone |
|---|---|
| Nov 8, 2026 | Baselines ready |
| Nov 22, 2026 | Models C and D running |
| Dec 13, 2026 | Trust evaluation complete |
| Dec 27, 2026 | Ablations and trade-offs complete |
| Jan 15, 2027 | Final report and code submission |
