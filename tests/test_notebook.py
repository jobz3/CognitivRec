"""Keep the notebook thin and free of outputs, so that it never causes merge conflicts."""

from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "CognitiveRec_GroupE.ipynb"


def load_notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_is_valid():
    nbformat.validate(load_notebook())


def test_notebook_has_no_committed_outputs():
    for cell in load_notebook().cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None


def test_notebook_runs_every_stage_and_defines_no_models():
    source = "\n".join(cell.source for cell in load_notebook().cells if cell.cell_type == "code")
    for stage in ("stage1_baselines", "stage2_cognitive", "stage3_trust", "stage4_ablation"):
        assert f"{stage}.run(" in source
    assert "class " not in source
