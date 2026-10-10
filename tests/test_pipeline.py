from pathlib import Path
import json

from scripts.make_sample_dataset import make_sample
from bpso_ids.config import make_config
from bpso_ids.pipeline import ResearchPipeline


def test_end_to_end_pipeline(tmp_path: Path):
    dataset = tmp_path / "sample.csv"
    make_sample(str(dataset), rows=240, features=8)
    cfg = make_config("demo")
    cfg.artifacts_dir = tmp_path / "artifacts"
    cfg.n_particles = 4
    cfg.n_iterations = 3
    cfg.demo_fitness_max_rows = 120
    cfg.max_rows = 240

    pipeline = ResearchPipeline(cfg)
    pipeline.load_and_prepare(dataset)
    pipeline.run_bpso()
    metrics = pipeline.train_models_and_fuzzy()
    pipeline.save_artifacts(metrics)

    assert pipeline.selected_features
    assert (cfg.artifacts_dir / "results.json").exists()
    payload = json.loads((cfg.artifacts_dir / "results.json").read_text())
    assert payload["total_features"] == 8
    assert "selected_test" in payload["metrics"]
