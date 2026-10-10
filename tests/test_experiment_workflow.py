from pathlib import Path

from scripts.make_sample_dataset import make_sample
from bpso_ids.config import make_config
from workflows.experiment import ExperimentWorkflow


def _demo_config(tmp_path: Path):
    """
    Small configuration so the workflow tests run quickly.
    """
    cfg = make_config("demo")

    cfg.artifacts_dir = tmp_path / "artifacts"

    cfg.n_particles = 3
    cfg.n_iterations = 2

    cfg.max_rows = 180
    cfg.demo_fitness_max_rows = 80
    cfg.demo_fitness_epochs = 2
    cfg.demo_model_epochs = 3

    return cfg


def test_workflow_load_and_analyze(tmp_path: Path):
    """
    Workflow should load a CSV and expose its dataset profile.
    """
    dataset = tmp_path / "sample.csv"

    make_sample(
        str(dataset),
        rows=120,
        features=6,
    )

    workflow = ExperimentWorkflow(
        _demo_config(tmp_path)
    )

    profile = workflow.load_csv(
        dataset,
        target_column="Label",
    )

    assert profile["target_column"] == "Label"
    assert profile["rows"] == 120
    assert workflow.dataset_profile is not None


def test_workflow_prepare_dataset(tmp_path: Path):
    """
    Workflow should expose a useful preprocessing summary.
    """
    dataset = tmp_path / "sample.csv"

    make_sample(
        str(dataset),
        rows=180,
        features=6,
    )

    workflow = ExperimentWorkflow(
        _demo_config(tmp_path)
    )

    workflow.load_csv(
        dataset,
        target_column="Label",
        drop_columns=["Scenario"],
    )

    prepared = workflow.prepare_dataset()

    assert prepared["target_column"] == "Label"

    assert prepared["train_rows"] > 0
    assert prepared["validation_rows"] > 0
    assert prepared["test_rows"] > 0

    assert prepared["features"] == 6


def test_workflow_bpso_stage(tmp_path: Path):
    """
    Workflow should successfully connect:
        preprocessing → baseline NN → BPSO
    """
    dataset = tmp_path / "sample.csv"

    make_sample(
        str(dataset),
        rows=180,
        features=6,
    )

    workflow = ExperimentWorkflow(
        _demo_config(tmp_path)
    )

    workflow.load_csv(
        dataset,
        target_column="Label",
        drop_columns=["Scenario"],
    )

    workflow.prepare_dataset()

    workflow.train_baseline()

    info = workflow.initialize_bpso()

    result = workflow.run_bpso()

    assert info["n_particles"] == 3
    assert info["n_iterations"] == 2

    assert result["selected_features"]

    assert len(
        result["history"]
    ) == 2

    assert "bpso" in (
        workflow.stage_status
    )


def test_workflow_run_all_and_export(
    tmp_path: Path,
):
    """
    Workflow should be able to execute the complete experiment
    and export its report.
    """
    dataset = tmp_path / "sample.csv"
    report_path = tmp_path / "report.json"

    make_sample(
        str(dataset),
        rows=180,
        features=6,
    )

    workflow = ExperimentWorkflow(
        _demo_config(tmp_path)
    )

    workflow.load_csv(
        dataset,
        target_column="Label",
        drop_columns=["Scenario"],
    )

    workflow.prepare_dataset()

    payload = workflow.run_all()

    assert "metrics" in payload

    assert payload[
        "selected_features"
    ]

    exported_path = workflow.export_report(
        report_path
    )

    assert Path(
        exported_path
    ) == report_path

    assert report_path.exists()