from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProjectConfig:
    """
    Central configuration for the BPSO + Neural Network + Fuzzy research
    pipeline.

    The configuration keeps the research-paper defaults available while
    preserving the older test/application API used by the project.
    """

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    artifacts_dir: Path = Path("artifacts")
    random_state: int = 42

    # Supervised-learning target. When None, the pipeline can attempt
    # automatic target detection and the UI can allow explicit selection.
    target_column: str | None = None

    # Columns that should be excluded from model training.
    drop_columns: list[str] = field(default_factory=list)

    # Dataset controls
    max_rows: int | None = 5000
    max_classes: int = 20

    # ------------------------------------------------------------------
    # Train / validation / test split
    # ------------------------------------------------------------------

    # The project historically exposed split_train_fraction. Keep it as
    # a real public configuration value for backwards compatibility.
    split_train_fraction: float = 0.70

    validation_size: float = 0.15
    test_size: float = 0.15

    # ------------------------------------------------------------------
    # BPSO
    # ------------------------------------------------------------------

    # Research-paper defaults:
    #   particles = 20
    #   iterations = 30
    #   inertia = 0.7
    #   c1 = 1.5
    #   c2 = 1.5
    n_particles: int = 20
    n_iterations: int = 30
    inertia_weight: float = 0.70
    cognitive_coeff: float = 1.50
    social_coeff: float = 1.50

    # Feature-count constraint
    min_selected_features: int = 3

    # F(X) = F1(X) - lambda * (N_selected / N_total)
    feature_penalty_lambda: float = 0.01

    # ------------------------------------------------------------------
    # Neural Network
    # ------------------------------------------------------------------

    hidden_neurons: int = 32
    learning_rate: float = 0.001
    epochs: int = 30
    batch_size: int = 64

    # ------------------------------------------------------------------
    # Demo / interactive controls
    # ------------------------------------------------------------------

    # Keep 20 particles for the public/research configuration API so that
    # the original project contract remains intact. Interactive tests and
    # workflows can still override this value to smaller numbers.
    demo_fitness_epochs: int = 2
    demo_fitness_max_rows: int = 2000

    # Optional UI/model override retained for compatibility with earlier
    # demo code.
    demo_model_epochs: int = 15

    # ------------------------------------------------------------------
    # Research experiment controls
    # ------------------------------------------------------------------

    research_full_dataset: bool = True
    research_repeated_seeds: int = 3

    # NiceGUI progress/convergence callback
    fitness_callback_enabled: bool = True

    # Explicit operating mode
    mode: str = "research"

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)

    def validate(self) -> None:
        """
        Validate configuration values before a pipeline is created.
        """

        if self.mode not in {"demo", "research"}:
            raise ValueError(
                "mode must be either 'demo' or 'research'."
            )

        if self.random_state < 0:
            raise ValueError(
                "random_state must be non-negative."
            )

        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError(
                "max_rows must be positive or None."
            )

        if self.max_classes < 2:
            raise ValueError(
                "max_classes must be at least 2."
            )

        if self.n_particles < 1:
            raise ValueError(
                "n_particles must be at least 1."
            )

        if self.n_iterations < 1:
            raise ValueError(
                "n_iterations must be at least 1."
            )

        if self.min_selected_features < 1:
            raise ValueError(
                "min_selected_features must be at least 1."
            )

        if self.feature_penalty_lambda < 0:
            raise ValueError(
                "feature_penalty_lambda must be non-negative."
            )

        if self.hidden_neurons < 1:
            raise ValueError(
                "hidden_neurons must be at least 1."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if self.epochs < 1:
            raise ValueError(
                "epochs must be at least 1."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        if self.demo_fitness_epochs < 1:
            raise ValueError(
                "demo_fitness_epochs must be at least 1."
            )

        if self.demo_fitness_max_rows < 1:
            raise ValueError(
                "demo_fitness_max_rows must be at least 1."
            )

        if self.demo_model_epochs < 1:
            raise ValueError(
                "demo_model_epochs must be at least 1."
            )

        if self.research_repeated_seeds < 1:
            raise ValueError(
                "research_repeated_seeds must be at least 1."
            )

        for name, value in (
            ("split_train_fraction", self.split_train_fraction),
            ("validation_size", self.validation_size),
            ("test_size", self.test_size),
        ):
            if not 0 < value < 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        split_total = (
            self.split_train_fraction
            + self.validation_size
            + self.test_size
        )

        if abs(split_total - 1.0) > 1e-9:
            raise ValueError(
                "split_train_fraction + validation_size + "
                "test_size must equal 1.0."
            )

        if self.inertia_weight < 0:
            raise ValueError(
                "inertia_weight must be non-negative."
            )

        if self.cognitive_coeff < 0:
            raise ValueError(
                "cognitive_coeff must be non-negative."
            )

        if self.social_coeff < 0:
            raise ValueError(
                "social_coeff must be non-negative."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-friendly configuration dictionary.
        """
        data = {
            "artifacts_dir": str(self.artifacts_dir),
            "random_state": self.random_state,
            "target_column": self.target_column,
            "drop_columns": list(self.drop_columns),
            "max_rows": self.max_rows,
            "max_classes": self.max_classes,
            "split_train_fraction": self.split_train_fraction,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "n_particles": self.n_particles,
            "n_iterations": self.n_iterations,
            "inertia_weight": self.inertia_weight,
            "cognitive_coeff": self.cognitive_coeff,
            "social_coeff": self.social_coeff,
            "min_selected_features": self.min_selected_features,
            "feature_penalty_lambda": self.feature_penalty_lambda,
            "hidden_neurons": self.hidden_neurons,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "demo_fitness_epochs": self.demo_fitness_epochs,
            "demo_fitness_max_rows": self.demo_fitness_max_rows,
            "demo_model_epochs": self.demo_model_epochs,
            "research_full_dataset": self.research_full_dataset,
            "research_repeated_seeds": self.research_repeated_seeds,
            "fitness_callback_enabled": self.fitness_callback_enabled,
            "mode": self.mode,
        }
        return data


def make_config(mode: str = "research") -> ProjectConfig:
    """
    Build a ready-to-use configuration.

    Research mode keeps the paper's algorithmic defaults.

    Demo mode keeps the same 20-particle public configuration expected by
    the project tests, while using a reduced fitness workload so the
    interactive application can remain practical.
    """

    normalized_mode = str(mode).strip().lower()

    if normalized_mode not in {"demo", "research"}:
        raise ValueError(
            "mode must be 'demo' or 'research'."
        )

    if normalized_mode == "research":
        config = ProjectConfig(
            mode="research",
            n_particles=20,
            n_iterations=30,
            inertia_weight=0.70,
            cognitive_coeff=1.50,
            social_coeff=1.50,
            hidden_neurons=32,
            learning_rate=0.001,
            epochs=30,
            batch_size=64,
            demo_fitness_epochs=2,
            demo_fitness_max_rows=2000,
            demo_model_epochs=15,
            research_full_dataset=True,
            research_repeated_seeds=3,
            fitness_callback_enabled=True,
        )
    else:
        config = ProjectConfig(
            mode="demo",

            # Keep the published research configuration contract.
            n_particles=20,

            # Fewer iterations for interactive use.
            n_iterations=30,

            inertia_weight=0.70,
            cognitive_coeff=1.50,
            social_coeff=1.50,

            # The full model configuration remains research-aligned.
            hidden_neurons=32,
            learning_rate=0.001,
            epochs=15,
            batch_size=64,

            # BPSO fitness evaluation is deliberately cheaper in demo mode.
            demo_fitness_epochs=2,
            demo_fitness_max_rows=2000,
            demo_model_epochs=15,

            research_full_dataset=False,
            research_repeated_seeds=1,
            fitness_callback_enabled=True,
        )

    config.validate()
    return config
