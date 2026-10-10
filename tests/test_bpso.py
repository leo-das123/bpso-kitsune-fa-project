import numpy as np

from bpso_ids.bpso import BinaryPSO


def test_sigmoid_range():
    values = BinaryPSO.sigmoid(np.array([-100.0, 0.0, 100.0]))
    assert np.all(values >= 0)
    assert np.all(values <= 1)


def test_bpso_finds_high_fitness_subset():
    def fitness(position):
        return 1.0 if position[0] and position[1] else 0.0

    opt = BinaryPSO(
        n_features=4,
        fitness_fn=fitness,
        n_particles=8,
        n_iterations=8,
        min_selected_features=1,
        random_state=4,
    )
    result = opt.optimize()
    assert result.best_fitness == 1.0
    assert result.best_position.shape == (4,)
    assert len(result.history) == 8
