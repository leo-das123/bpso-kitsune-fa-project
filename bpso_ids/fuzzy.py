from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np


def triangular_membership(
    x: float,
    a: float,
    b: float,
    c: float,
) -> float:
    """Triangular membership function with shoulder handling."""
    x = float(x)
    a = float(a)
    b = float(b)
    c = float(c)

    if not (a <= b <= c):
        raise ValueError(
            "Triangular membership parameters must satisfy a <= b <= c."
        )

    if a == b == c:
        return 1.0 if x == b else 0.0

    # Left shoulder.
    if a == b:
        if x <= b:
            return 1.0
        if x >= c:
            return 0.0
        return float((c - x) / (c - b))

    # Right shoulder.
    if b == c:
        if x >= b:
            return 1.0
        if x <= a:
            return 0.0
        return float((x - a) / (b - a))

    if x <= a or x >= c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        return float((x - a) / (b - a))

    return float((c - x) / (c - b))


@dataclass
class FuzzyRule:
    """Interpretable AND-rule for the compact FIS."""

    antecedents: dict[str, str]
    consequent: str
    weight: float = 1.0
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FuzzyConfig:
    """
    Configuration for the three-input interpretable FIS.

    The feature names are supplied by the experiment/UI. No Kitsune-specific
    feature names are hard-coded here.
    """

    feature_names: list[str]

    low: tuple[float, float, float] = (
        0.0,
        0.0,
        0.5,
    )

    medium: tuple[float, float, float] = (
        0.25,
        0.5,
        0.75,
    )

    high: tuple[float, float, float] = (
        0.5,
        1.0,
        1.0,
    )

    rules: list[FuzzyRule] = field(
        default_factory=list
    )

    output_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Low": 0.17,
            "Medium": 0.50,
            "High": 0.83,
        }
    )

    def __post_init__(self) -> None:
        self.feature_names = [
            str(name)
            for name in self.feature_names
        ]

        if len(self.feature_names) != 3:
            raise ValueError(
                "FuzzyConfig requires exactly three input features."
            )

        if len(set(self.feature_names)) != 3:
            raise ValueError(
                "Fuzzy input feature names must be unique."
            )

        if not self.rules:
            self.rules = default_rules(
                self.feature_names
            )

        required_outputs = {
            "Low",
            "Medium",
            "High",
        }

        for rule in self.rules:
            if rule.consequent not in required_outputs:
                raise ValueError(
                    f"Unsupported fuzzy consequent: {rule.consequent}"
                )

            unknown = set(
                rule.antecedents
            ) - set(self.feature_names)

            if unknown:
                raise ValueError(
                    "Fuzzy rule contains unknown input features: "
                    f"{sorted(unknown)}"
                )

        for name in required_outputs:
            if name not in self.output_weights:
                raise ValueError(
                    f"Missing output weight for {name}."
                )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["low"] = list(self.low)
        out["medium"] = list(self.medium)
        out["high"] = list(self.high)
        out["rules"] = [
            rule.to_dict()
            for rule in self.rules
        ]
        return out

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> "FuzzyConfig":
        data = dict(values)

        data["low"] = tuple(
            data.get(
                "low",
                (0.0, 0.0, 0.5),
            )
        )
        data["medium"] = tuple(
            data.get(
                "medium",
                (0.25, 0.5, 0.75),
            )
        )
        data["high"] = tuple(
            data.get(
                "high",
                (0.5, 1.0, 1.0),
            )
        )

        data["rules"] = [
            FuzzyRule(**rule)
            if not isinstance(rule, FuzzyRule)
            else rule
            for rule in data.get(
                "rules",
                [],
            )
        ]

        return cls(
            **data
        )


def default_rules(
    feature_names: Iterable[str],
) -> list[FuzzyRule]:
    """
    Create generic rules based only on input positions.

    These are engineering defaults. The experiment may replace them with
    calibrated rules after inspecting the selected features.
    """
    names = list(feature_names)
    if len(names) != 3:
        raise ValueError(
            "Three feature names are required."
        )

    a, b, c = names

    return [
        FuzzyRule(
            name="all_low",
            antecedents={
                a: "Low",
                b: "Low",
                c: "Low",
            },
            consequent="Low",
        ),
        FuzzyRule(
            name="mixed_medium",
            antecedents={
                a: "Medium",
                b: "Medium",
                c: "Medium",
            },
            consequent="Medium",
        ),
        FuzzyRule(
            name="all_high",
            antecedents={
                a: "High",
                b: "High",
                c: "High",
            },
            consequent="High",
        ),
        FuzzyRule(
            name="two_high",
            antecedents={
                a: "High",
                b: "High",
            },
            consequent="High",
        ),
        FuzzyRule(
            name="high_and_medium",
            antecedents={
                a: "High",
                b: "Medium",
            },
            consequent="Medium",
        ),
        FuzzyRule(
            name="mostly_low",
            antecedents={
                a: "Low",
                b: "Low",
                c: "Medium",
            },
            consequent="Low",
        ),
    ]


class FuzzyRiskEngine:
    """
    Compact, interpretable fuzzy inference engine.

    Inputs are normalized to [0, 1] before inference. The pipeline is
    responsible for converting raw/scaled feature values into that range.
    """

    LINGUISTIC_TERMS = (
        "Low",
        "Medium",
        "High",
    )

    def __init__(
        self,
        config: FuzzyConfig,
    ) -> None:
        self.config = config

    def memberships_for_value(
        self,
        value: float,
    ) -> dict[str, float]:
        value = float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

        return {
            "Low": triangular_membership(
                value,
                *self.config.low,
            ),
            "Medium": triangular_membership(
                value,
                *self.config.medium,
            ),
            "High": triangular_membership(
                value,
                *self.config.high,
            ),
        }

    def _rule_strength(
        self,
        rule: FuzzyRule,
        memberships: dict[str, dict[str, float]],
    ) -> float:
        strengths = []

        for feature_name, linguistic_term in (
            rule.antecedents.items()
        ):
            strengths.append(
                memberships[feature_name][
                    linguistic_term
                ]
            )

        if not strengths:
            return 0.0

        return float(
            min(strengths)
            * max(0.0, float(rule.weight))
        )

    def infer(
        self,
        values: list[float] | np.ndarray,
    ) -> dict[str, Any]:
        values = np.asarray(
            values,
            dtype=float,
        )

        if values.shape != (3,):
            raise ValueError(
                "Fuzzy inference expects exactly three values."
            )

        normalized = np.clip(
            np.nan_to_num(
                values,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )

        memberships = {
            feature_name: self.memberships_for_value(value)
            for feature_name, value in zip(
                self.config.feature_names,
                normalized,
            )
        }

        rule_strengths: dict[str, float] = {}
        aggregated = {
            "Low": 0.0,
            "Medium": 0.0,
            "High": 0.0,
        }

        for index, rule in enumerate(
            self.config.rules,
            start=1,
        ):
            strength = self._rule_strength(
                rule,
                memberships,
            )
            name = (
                rule.name
                if rule.name
                else f"rule_{index}"
            )
            rule_strengths[name] = strength
            aggregated[rule.consequent] = max(
                aggregated[rule.consequent],
                strength,
            )

        denominator = sum(
            aggregated.values()
        )

        if denominator > 0:
            risk_score = float(
                sum(
                    aggregated[level]
                    * self.config.output_weights[level]
                    for level in self.LINGUISTIC_TERMS
                )
                / denominator
            )
        else:
            # Safe fallback if no rule fires.
            risk_score = float(
                np.mean(normalized)
            )

        if risk_score < 1.0 / 3.0:
            risk_level = "Low"
        elif risk_score < 2.0 / 3.0:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "inputs": {
                name: float(value)
                for name, value in zip(
                    self.config.feature_names,
                    normalized,
                )
            },
            "memberships": memberships,
            "rule_strengths": rule_strengths,
            "aggregated_risk": aggregated,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "rules": [
                rule.to_dict()
                for rule in self.config.rules
            ],
        }

    def explain(
        self,
        result: dict[str, Any],
    ) -> str:
        """Create a concise human-readable explanation for the UI."""
        level = result["risk_level"]
        score = result["risk_score"]

        active_rules = sorted(
            result["rule_strengths"].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        active = [
            name
            for name, strength in active_rules
            if strength > 0
        ]

        if active:
            return (
                f"Risk is {level.upper()} with score {score:.3f}. "
                f"Active rules: {', '.join(active[:3])}."
            )

        return (
            f"Risk is {level.upper()} with score {score:.3f}. "
            "No fuzzy rule fired strongly, so the system used the "
            "normalized-input fallback.")
