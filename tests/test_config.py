from bpso_ids.config import make_config


def test_demo_config_is_valid():
    cfg = make_config("demo")
    assert cfg.n_particles == 20
    assert cfg.n_iterations == 30
    assert cfg.hidden_neurons == 32


def test_split_fractions_sum_to_one():
    cfg = make_config("research")
    assert abs(cfg.split_train_fraction + cfg.validation_size + cfg.test_size - 1.0) < 1e-9
