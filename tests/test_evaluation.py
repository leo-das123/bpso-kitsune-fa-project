import numpy as np

from bpso_ids.neural_network import FixedNeuralNetwork
from bpso_ids.evaluation import evaluate_model


def test_evaluation_metrics_are_valid():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10, dtype=float)
    y = np.array([0, 0, 1, 1] * 10)
    model = FixedNeuralNetwork(hidden_neurons=8, epochs=20, batch_size=8, random_state=3)
    model.fit(X, y)
    result = evaluate_model(model, X, y)
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1
    assert 0 <= result.f1 <= 1
