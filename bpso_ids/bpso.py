from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


FitnessReturn = float | tuple[float, dict[str, Any]]
FitnessFunction = Callable[[np.ndarray], FitnessReturn]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class Particle:
    """One binary PSO particle."""

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float = float("-inf")
    best_details: dict[str, Any] | None = None


@dataclass
class BPSOResult:
    """Complete BPSO optimization result."""

    best_position: np.ndarray
    best_fitness: float
    best_details: dict[str, Any]
    history: list[dict[str, float]]
    final_particles: list[dict[str, Any]]
    final_state: dict[str, Any]

    @property
    def selected_count(self) -> int:
        return int(self.best_position.sum())


class BinaryPSO:
    """
    Binary Particle Swarm Optimization for wrapper-based feature selection.

    The optimizer is deliberately model-agnostic. A fitness function receives
    one binary feature mask and can return either:

        fitness

    or:

        (fitness, details)

    The details object is retained for the current global-best solution and is
    exposed to the application so the UI can display validation F1, selected
    feature count, penalty and final fitness.
    """

    def __init__(
        self,
        n_features: int,
        fitness_fn: FitnessFunction,
        n_particles: int = 20,
        n_iterations: int = 30,
        inertia_weight: float = 0.7,
        cognitive_coeff: float = 1.5,
        social_coeff: float = 1.5,
        min_selected_features: int = 2,
        random_state: int = 42,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if n_particles < 2:
            raise ValueError("n_particles must be >= 2")
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if inertia_weight < 0:
            raise ValueError("inertia_weight must be >= 0")
        if cognitive_coeff < 0:
            raise ValueError("cognitive_coeff must be >= 0")
        if social_coeff < 0:
            raise ValueError("social_coeff must be >= 0")

        self.n_features = int(n_features)
        self.fitness_fn = fitness_fn
        self.n_particles = int(n_particles)
        self.n_iterations = int(n_iterations)
        self.w = float(inertia_weight)
        self.c1 = float(cognitive_coeff)
        self.c2 = float(social_coeff)
        self.min_selected_features = min(
            max(1, int(min_selected_features)),
            self.n_features,
        )
        self.rng = np.random.default_rng(random_state)
        self.random_state = int(random_state)
        self.progress_callback = progress_callback

        self.swarm: list[Particle] = []
        for _ in range(self.n_particles):
            position = self.rng.integers(
                0,
                2,
                size=self.n_features,
                dtype=np.int8,
            ).astype(bool)
            self._repair(position)

            velocity = self.rng.normal(
                loc=0.0,
                scale=1.0,
                size=self.n_features,
            )

            self.swarm.append(
                Particle(
                    position=position,
                    velocity=velocity,
                    best_position=position.copy(),
                )
            )

        self.global_best_position = self.swarm[0].position.copy()
        self.global_best_fitness = float("-inf")
        self.global_best_details: dict[str, Any] = {}

        self.last_state: dict[str, Any] = {}
        self._fitness_details: dict[str, Any] = {}
        self.current_iteration = 0

    # ------------------------------------------------------------------
    # Core PSO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Map continuous velocity to a probability in [0, 1]."""
        x = np.clip(x, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _repair(self, position: np.ndarray) -> None:
        """Ensure every particle contains the minimum number of features."""
        selected = int(position.sum())
        if selected >= self.min_selected_features:
            return

        required = self.min_selected_features - selected
        zero_indices = np.flatnonzero(~position)

        if required > 0 and len(zero_indices) > 0:
            chosen = self.rng.choice(
                zero_indices,
                size=required,
                replace=False,
            )
            position[chosen] = True

    def _evaluate_fitness(
        self,
        position: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        """Normalize fitness callback output to (score, details)."""
        if int(position.sum()) < self.min_selected_features:
            return (
                float("-inf"),
                {
                    "validation_f1": float("nan"),
                    "selected_features": int(position.sum()),
                    "total_features": self.n_features,
                    "feature_penalty": float("nan"),
                    "fitness": float("-inf"),
                },
            )

        result = self.fitness_fn(position.copy())

        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError(
                    "Fitness function tuple result must be "
                    "(fitness, details)."
                )
            score = float(result[0])
            details = dict(result[1] or {})
        else:
            score = float(result)
            details = {}

        details.setdefault(
            "selected_features",
            int(position.sum()),
        )
        details.setdefault(
            "total_features",
            self.n_features,
        )
        details["fitness"] = score

        return score, details

    def _evaluate_particle(
        self,
        particle: Particle,
    ) -> float:
        score, details = self._evaluate_fitness(
            particle.position
        )

        if score > particle.best_fitness:
            particle.best_fitness = score
            particle.best_position = particle.position.copy()
            particle.best_details = dict(details)

        if score > self.global_best_fitness:
            self.global_best_fitness = score
            self.global_best_position = particle.position.copy()
            self.global_best_details = dict(details)

        return score

    def _snapshot(
        self,
        fitness_values: list[float],
        iteration: int,
        phase: str = "iteration",
    ) -> dict[str, Any]:
        valid = [
            value
            for value in fitness_values
            if np.isfinite(value)
        ]

        particles = []
        for index, particle in enumerate(
            self.swarm,
            start=1,
        ):
            particles.append(
                {
                    "index": index,
                    "position": particle.position.astype(int).tolist(),
                    "velocity": [
                        float(value)
                        for value in particle.velocity
                    ],
                    "fitness": float(particle.best_fitness),
                    "pbest": particle.best_position.astype(int).tolist(),
                    "pbest_fitness": float(particle.best_fitness),
                    "pbest_details": dict(
                        particle.best_details or {}
                    ),
                }
            )

        return {
            "phase": phase,
            "iteration": int(iteration),
            "total_iterations": int(self.n_iterations),
            "best_fitness": float(self.global_best_fitness),
            "mean_fitness": (
                float(np.mean(valid))
                if valid
                else float("-inf")
            ),
            "selected_features": int(
                self.global_best_position.sum()
            ),
            "particles": particles,
            "gbest": self.global_best_position.astype(int).tolist(),
            "gbest_fitness": float(self.global_best_fitness),
            "gbest_details": dict(
                self.global_best_details
            ),
        }

    def _emit(self, state: dict[str, Any]) -> None:
        self.last_state = state
        if self.progress_callback is not None:
            self.progress_callback(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> BPSOResult:
        history: list[dict[str, float]] = []

        for iteration in range(
            1,
            self.n_iterations + 1,
        ):
            self.current_iteration = iteration

            fitness_values = [
                self._evaluate_particle(
                    particle
                )
                for particle in self.swarm
            ]

            state = self._snapshot(
                fitness_values,
                iteration,
                phase="iteration",
            )

            details = self.fitness_details()
            self._fitness_details = details

            history.append(
                {
                    "iteration": float(iteration),
                    "best_fitness": float(
                        self.global_best_fitness
                    ),
                    "mean_fitness": float(
                        state["mean_fitness"]
                    ),
                    "selected_features": float(
                        self.global_best_position.sum()
                    ),
                    "validation_f1": float(
                        details.get(
                            "validation_f1",
                            np.nan,
                        )
                    ),
                    "feature_penalty": float(
                        details.get(
                            "feature_penalty",
                            np.nan,
                        )
                    ),
                }
            )

            # Emit before the swarm moves so the UI can display the actual
            # particle state corresponding to the current fitness evaluation.
            state["fitness_details"] = details
            self._emit(state)

            # No position update is required after the final evaluation.
            if iteration == self.n_iterations:
                break

            for particle in self.swarm:
                r1 = self.rng.random(
                    self.n_features
                )
                r2 = self.rng.random(
                    self.n_features
                )

                cognitive = (
                    self.c1
                    * r1
                    * (
                        particle.best_position.astype(float)
                        - particle.position.astype(float)
                    )
                )

                social = (
                    self.c2
                    * r2
                    * (
                        self.global_best_position.astype(float)
                        - particle.position.astype(float)
                    )
                )

                particle.velocity = (
                    self.w * particle.velocity
                    + cognitive
                    + social
                )

                probability = self.sigmoid(
                    particle.velocity
                )

                particle.position = (
                    self.rng.random(
                        self.n_features
                    )
                    < probability
                )

                self._repair(
                    particle.position
                )

        final_state = self._snapshot(
            [
                particle.best_fitness
                for particle in self.swarm
            ],
            self.current_iteration,
            phase="complete",
        )
        final_state["fitness_details"] = self.fitness_details()
        self.last_state = final_state

        return BPSOResult(
            best_position=self.global_best_position.copy(),
            best_fitness=float(
                self.global_best_fitness
            ),
            best_details=self.fitness_details(),
            history=history,
            final_particles=list(
                final_state["particles"]
            ),
            final_state=final_state,
        )

    def fitness_details(self) -> dict[str, Any]:
        """Return details for the current global-best solution."""
        details = dict(
            self.global_best_details
        )
        details.setdefault(
            "selected_features",
            int(self.global_best_position.sum()),
        )
        details.setdefault(
            "total_features",
            int(self.n_features),
        )
        details.setdefault(
            "fitness",
            float(self.global_best_fitness),
        )
        return details

    @property
    def global_best(self) -> np.ndarray:
        """Compatibility alias for the current swarm best position."""
        return self.global_best_position
