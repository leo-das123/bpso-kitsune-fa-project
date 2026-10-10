import numpy as np

from bpso_ids.fuzzy import triangular_membership, FuzzyConfig, FuzzyRiskEngine


def test_triangular_membership():
    assert triangular_membership(0.5, 0.0, 0.5, 1.0) == 1.0
    assert triangular_membership(-1.0, 0.0, 0.5, 1.0) == 0.0


def test_fuzzy_high_risk_for_high_inputs():
    engine = FuzzyRiskEngine(FuzzyConfig(["F1", "F2", "F3"]))
    result = engine.infer([0.95, 0.98, 0.92])
    assert result["risk_level"] == "High"
    assert result["risk_score"] > 0.67


def test_fuzzy_low_risk_for_low_inputs():
    engine = FuzzyRiskEngine(FuzzyConfig(["F1", "F2", "F3"]))
    result = engine.infer([0.05, 0.1, 0.08])
    assert result["risk_level"] == "Low"
