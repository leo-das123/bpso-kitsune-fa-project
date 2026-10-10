from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from nicegui import ui

from ui.components.sidebar import NAV_ITEMS


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric(source: dict[str, Any] | None, name: str) -> float | None:
    if not source:
        return None
    value = source.get(name)
    if value is None:
        return None
    return _safe_float(value)


def _status_icon(status: str) -> str:
    return {
        "complete": "✅",
        "configured": "🟡",
        "running": "🔵",
        "pending": "⚪",
        "error": "❌",
    }.get(status, "⚪")


def _metric_card(
    title: str,
    value: Any,
    caption: str = "",
) -> None:
    with ui.card().classes(
        "flex-1 min-w-[180px] p-4"
    ):
        ui.label(title).classes(
            "text-xs text-gray-500"
        )
        ui.label(_fmt(value, 3) if isinstance(value, float) else str(value)).classes(
            "text-2xl font-bold"
        )
        if caption:
            ui.label(caption).classes(
                "text-xs text-gray-500"
            )


def _section_title(
    title: str,
    subtitle: str = "",
) -> None:
    ui.label(title).classes(
        "text-2xl font-bold"
    )
    if subtitle:
        ui.label(subtitle).classes(
            "text-sm text-gray-500 mb-2"
        )


def _key_value(
    label: str,
    value: Any,
) -> None:
    with ui.row().classes(
        "w-full justify-between items-center py-1"
    ):
        ui.label(label).classes(
            "text-sm text-gray-600"
        )
        ui.label(_fmt(value)).classes(
            "text-sm font-medium"
        )


def _status_chips(state: Any) -> None:
    statuses = state.workflow.stage_status
    with ui.row().classes(
        "w-full gap-2 flex-wrap"
    ):
        for stage, status in statuses.items():
            ui.chip(
                f"{stage.replace('_', ' ').title()}: {status}",
                icon=_status_icon(status),
            )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def _render_dashboard(state: Any) -> None:
    _section_title(
        "BPSO-IntrusionGuard",
        "Research workflow dashboard",
    )

    ui.label(
        "More features → BPSO → fewer useful features → neural-network classification → fuzzy risk interpretation"
    ).classes(
        "text-sm text-gray-600 mb-4"
    )

    with ui.row().classes(
        "w-full gap-4 flex-wrap"
    ):
        profile = state.workflow.dataset_profile or {}
        prepared = getattr(
            state.workflow.pipeline,
            "prepared",
            None,
        )
        comparison = getattr(
            state.workflow.pipeline,
            "comparison",
            None,
        )

        total_features = (
            profile.get("numeric_count", 0)
            + profile.get("categorical_count", 0)
            if profile
            else 0
        )

        if prepared is not None:
            total_features = len(
                prepared.feature_names
            )

        selected_count = len(
            state.workflow.selected_features
        )

        reduction = None
        if total_features and selected_count:
            reduction = 100.0 * (
                1.0 - selected_count / total_features
            )

        baseline_f1 = None
        selected_f1 = None
        if comparison is not None:
            baseline_f1 = getattr(
                comparison.baseline,
                "f1",
                None,
            )
            selected_f1 = getattr(
                comparison.bpso,
                "f1",
                None,
            )

        _metric_card(
            "Dataset",
            profile.get("rows", "—") if profile else "Not loaded",
            "rows",
        )
        _metric_card(
            "Usable Features",
            total_features or "—",
            "after preprocessing",
        )
        _metric_card(
            "Selected Features",
            selected_count or "—",
            "BPSO",
        )
        _metric_card(
            "Reduction",
            f"{reduction:.2f}%" if reduction is not None else "—",
            "feature reduction",
        )
        _metric_card(
            "Baseline F1",
            f"{baseline_f1:.4f}" if baseline_f1 is not None else "—",
            "all-feature NN",
        )
        _metric_card(
            "BPSO Model F1",
            f"{selected_f1:.4f}" if selected_f1 is not None else "—",
            "selected-feature NN",
        )

    with ui.card().classes(
        "w-full mt-4 p-5"
    ):
        ui.label(
            "Research Pipeline"
        ).classes(
            "text-lg font-semibold mb-3"
        )

        with ui.row().classes(
            "w-full items-center justify-around gap-2 flex-wrap"
        ):
            for label in [
                "Dataset",
                "Preprocessing",
                "BPSO",
                "Selected NN",
                "Fuzzy Risk",
                "Evaluation",
            ]:
                ui.badge(
                    label,
                    color="primary",
                )
                if label != "Evaluation":
                    ui.icon("arrow_forward").classes(
                        "text-gray-400"
                    )

    progress = state.progress or {}
    bpso_details = (
        progress.get("fitness_details", {})
        if isinstance(progress, dict)
        else {}
    )

    with ui.card().classes(
        "w-full mt-4 p-5"
    ):
        ui.label(
            "Live Experiment Monitor"
        ).classes(
            "text-lg font-semibold mb-3"
        )

        if state.running or progress:
            iteration = progress.get("iteration")
            total_iterations = progress.get("total_iterations")
            best_fitness = progress.get(
                "best_fitness",
                bpso_details.get("fitness"),
            )
            selected = progress.get(
                "selected_features",
                bpso_details.get("selected_features"),
            )

            with ui.row().classes(
                "w-full gap-4 flex-wrap"
            ):
                _metric_card(
                    "Stage",
                    state.running_stage or progress.get("phase", "—"),
                )
                _metric_card(
                    "Iteration",
                    (
                        f"{int(iteration)} / {int(total_iterations)}"
                        if iteration is not None and total_iterations
                        else "—"
                    ),
                )
                _metric_card(
                    "Best Fitness",
                    _fmt(best_fitness, 6),
                )
                _metric_card(
                    "Selected",
                    selected if selected is not None else "—",
                )

            if (
                iteration is not None
                and total_iterations
            ):
                ui.linear_progress(
                    min(
                        max(
                            _safe_float(iteration)
                            / max(
                                _safe_float(total_iterations),
                                1.0,
                            ),
                            0.0,
                        ),
                        1.0,
                    )
                ).classes("w-full mt-3")
        else:
            ui.label(
                "Run the experiment or BPSO to see live optimization state here."
            ).classes(
                "text-sm text-gray-500"
            )

    with ui.card().classes(
        "w-full mt-4 p-5"
    ):
        ui.label(
            "Experiment Status"
        ).classes(
            "text-lg font-semibold mb-3"
        )
        _status_chips(state)

    with ui.row().classes(
        "w-full mt-4 gap-3 flex-wrap"
    ):
        ui.button(
            "Run Complete Experiment",
            icon="play_arrow",
            on_click=state.run_all,
        ).props(
            "unelevated no-caps"
        ).classes(
            "bg-blue-600 text-white"
        )
        ui.button(
            "Go to Dataset",
            icon="upload_file",
            on_click=lambda: state.change_page("dataset"),
        ).props("outline no-caps")
        ui.button(
            "View Comparison",
            icon="compare_arrows",
            on_click=lambda: state.change_page("comparison"),
        ).props("outline no-caps")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _render_dataset(state: Any) -> None:
    _section_title(
        "Dataset",
        "Upload or select a labeled CSV, preview it, choose the target, then explicitly load it into the research workflow.",
    )

    with ui.card().classes("w-full p-5"):
        ui.label("Upload Network Dataset").classes(
            "text-lg font-semibold"
        )
        ui.label(
            "Uploading validates the CSV only. BPSO never starts automatically."
        ).classes(
            "text-sm text-gray-500 mb-3"
        )

        ui.upload(
            label="Upload CSV",
            auto_upload=True,
            on_upload=state.handle_upload,
        ).props("accept=.csv").classes("w-full")

        with ui.row().classes(
            "w-full items-end gap-3 mt-4"
        ):
            path_input = ui.input(
                "Existing CSV path",
                value=state.pending_dataset_path or state.dataset_path,
            ).classes("flex-1")

            ui.button(
                "Preview",
                icon="visibility",
                on_click=lambda: state.load_existing_for_preview(
                    str(path_input.value or "")
                ),
            ).props("outline no-caps")

    if state.preview_error:
        ui.label(
            f"CSV validation error: {state.preview_error}"
        ).classes(
            "text-sm text-red-600 mt-3"
        )

    if state.preview_df is not None:
        preview_df = state.preview_df
        columns = state.preview_columns

        with ui.card().classes("w-full mt-4 p-5"):
            ui.label("File Validation").classes(
                "text-lg font-semibold mb-2"
            )

            with ui.row().classes("gap-2 flex-wrap"):
                ui.chip(
                    "CSV readable",
                    icon="check_circle",
                ).classes("bg-green-50")
                ui.chip(
                    f"{len(columns)} columns detected",
                    icon="table_view",
                )
                ui.chip(
                    "Preview loaded",
                    icon="visibility",
                )

            ui.label(
                "Preview"
            ).classes(
                "text-lg font-semibold mt-4 mb-2"
            )

            ui.table(
                columns=[
                    {
                        "name": str(column),
                        "label": str(column),
                        "field": str(column),
                    }
                    for column in preview_df.columns
                ],
                rows=[
                    {
                        str(key): (
                            None if pd.isna(value) else value
                        )
                        for key, value in row.items()
                    }
                    for row in preview_df.to_dict(
                        orient="records"
                    )
                ],
            ).props(
                "dense bordered wrap-cells"
            ).classes("w-full")

            ui.label(
                "Target column"
            ).classes(
                "text-sm font-semibold mt-4"
            )

            ui.select(
                columns,
                value=(
                    state.target_column
                    if state.target_column in columns
                    else (
                        columns[-1]
                        if columns
                        else None
                    )
                ),
                label="Choose supervised classification target",
                on_change=lambda event: state.set_target(event.value),
            ).classes("w-full")

            ui.button(
                "Load Dataset",
                icon="folder_open",
                on_click=state.load_dataset,
            ).props(
                "unelevated no-caps"
            ).classes(
                "mt-4 bg-blue-600 text-white"
            )

    profile = state.workflow.dataset_profile

    if profile:
        with ui.card().classes("w-full mt-4 p-5"):
            ui.label("Dataset Information").classes(
                "text-lg font-semibold mb-3"
            )

            with ui.row().classes(
                "w-full gap-4 flex-wrap"
            ):
                _metric_card("Rows", profile.get("rows", "—"))
                _metric_card("Columns", profile.get("columns", "—"))
                _metric_card("Numeric", profile.get("numeric_count", "—"))
                _metric_card("Categorical", profile.get("categorical_count", "—"))
                _metric_card("Missing Cells", profile.get("total_missing_cells", "—"))
                _metric_card("Classes", profile.get("target_class_count", "—"))

        with ui.card().classes("w-full mt-4 p-5"):
            ui.label("Target Analysis").classes(
                "text-lg font-semibold mb-2"
            )

            _key_value(
                "Selected target",
                profile.get("target_column") or state.target_column or "—",
            )
            _key_value(
                "Suggested targets",
                ", ".join(
                    profile.get(
                        "possible_target_columns",
                        [],
                    )
                ) or "None detected",
            )

            distribution = profile.get(
                "class_distribution",
                {},
            )

            if distribution:
                ui.label("Class distribution").classes(
                    "font-medium mt-3"
                )
                for key, value in distribution.items():
                    with ui.row().classes(
                        "w-full justify-between py-1"
                    ):
                        ui.label(str(key))
                        ui.label(f"{value:,}").classes(
                            "font-medium"
                        )

        with ui.row().classes("w-full mt-4 gap-3"):
            if state.workflow.stage_status.get("preprocessing") == "complete":
                ui.chip(
                    "Ready for BPSO",
                    icon="check_circle",
                )
            else:
                ui.button(
                    "Prepare Dataset",
                    icon="tune",
                    on_click=lambda: state.change_page("preprocessing"),
                ).props("outline no-caps")
    else:
        ui.label(
            "Upload or preview a CSV to begin."
        ).classes(
            "text-sm text-gray-500 mt-4"
        )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _render_preprocessing(state: Any) -> None:
    _section_title(
        "Preprocessing",
        "Leakage-safe preparation: target handling, metadata removal, imputation, encoding, scaling, and train/validation/test split.",
    )

    profile = state.workflow.dataset_profile
    if not profile:
        ui.label(
            "Load a dataset first."
        ).classes(
            "text-sm text-gray-500"
        )
        return

    raw_df = getattr(
        state.workflow.pipeline,
        "raw_dataframe",
        None,
    )

    columns = (
        [str(col) for col in raw_df.columns]
        if raw_df is not None
        else []
    )

    target = state.target_column or profile.get(
        "target_column"
    )

    with ui.card().classes(
        "w-full p-5"
    ):
        ui.label(
            "Preprocessing Controls"
        ).classes(
            "text-lg font-semibold mb-3"
        )

        ui.select(
            columns,
            value=target,
            label="Target column",
            on_change=lambda e: state.set_target(
                e.value
            ),
        ).classes(
            "w-full"
        )

        suggested_drop = [
            str(x)
            for x in (
                profile.get(
                    "possible_id_columns",
                    [],
                )
                + profile.get(
                    "datetime_columns",
                    [],
                )
            )
        ]

        state.drop_columns = list(
            dict.fromkeys(
                state.drop_columns
                + suggested_drop
            )
        )

        ui.select(
            columns,
            multiple=True,
            value=[
                x for x in state.drop_columns
                if x in columns
            ],
            label="Optional columns to drop",
            on_change=lambda e: state.set_drop_columns(
                e.value
            ),
        ).props(
            "use-chips"
        ).classes(
            "w-full mt-3"
        )

        ui.button(
            "Run Preprocessing",
            icon="play_arrow",
            on_click=state.prepare_dataset,
        ).props(
            "unelevated no-caps"
        ).classes(
            "mt-4"
        )

    prepared = getattr(
        state.workflow.pipeline,
        "prepared",
        None,
    )

    if prepared is not None:
        with ui.row().classes(
            "w-full mt-4 gap-4 flex-wrap"
        ):
            _metric_card(
                "Target",
                prepared.target_column,
            )
            _metric_card(
                "Training Rows",
                len(prepared.X_train),
            )
            _metric_card(
                "Validation Rows",
                len(prepared.X_val),
            )
            _metric_card(
                "Test Rows",
                len(prepared.X_test),
            )
            _metric_card(
                "Usable Features",
                len(prepared.feature_names),
            )
            _metric_card(
                "Classes",
                len(prepared.class_names),
            )

        with ui.card().classes(
            "w-full mt-4 p-5"
        ):
            ui.label(
                "Preparation Summary"
            ).classes(
                "text-lg font-semibold mb-2"
            )

            _key_value(
                "Missing cells before preprocessing",
                profile.get(
                    "total_missing_cells",
                    0,
                ),
            )
            _key_value(
                "Duplicate rows",
                profile.get(
                    "duplicate_rows",
                    0,
                ),
            )
            _key_value(
                "Dropped columns",
                ", ".join(
                    getattr(
                        state.workflow.pipeline.preprocessor,
                        "dropped_columns",
                        [],
                    )
                )
                or "None",
            )
            _key_value(
                "Feature transformation",
                "Median imputation + scaling for numeric data; encoding for categorical data",
            )


# ---------------------------------------------------------------------------
# BPSO
# ---------------------------------------------------------------------------

def _render_bpso(state: Any) -> None:
    _section_title(
        "BPSO Feature Selection",
        "The research core: binary particles represent candidate feature subsets and optimize validation F1 with a feature-count penalty.",
    )

    prepared = getattr(
        state.workflow.pipeline,
        "prepared",
        None,
    )

    if prepared is None:
        ui.label(
            "Prepare the dataset before running BPSO."
        ).classes(
            "text-sm text-gray-500"
        )
        return

    config = state.workflow.config

    with ui.card().classes(
        "w-full p-5"
    ):
        ui.label(
            "BPSO Configuration"
        ).classes(
            "text-lg font-semibold mb-3"
        )

        with ui.row().classes(
            "w-full gap-3 flex-wrap"
        ):
            particles = ui.number(
                "Particles",
                value=config.n_particles,
                min=2,
                max=200,
                step=1,
            ).classes(
                "flex-1 min-w-[160px]"
            )

            iterations = ui.number(
                "Iterations",
                value=config.n_iterations,
                min=1,
                max=200,
                step=1,
            ).classes(
                "flex-1 min-w-[160px]"
            )

            inertia = ui.number(
                "Inertia Weight (w)",
                value=config.inertia_weight,
                min=0,
                step=0.05,
            ).classes(
                "flex-1 min-w-[160px]"
            )

            c1 = ui.number(
                "Cognitive (c1)",
                value=config.cognitive_coeff,
                min=0,
                step=0.1,
            ).classes(
                "flex-1 min-w-[160px]"
            )

            c2 = ui.number(
                "Social (c2)",
                value=config.social_coeff,
                min=0,
                step=0.1,
            ).classes(
                "flex-1 min-w-[160px]"
            )

            penalty = ui.number(
                "Penalty λ",
                value=config.feature_penalty_lambda,
                min=0,
                step=0.001,
            ).classes(
                "flex-1 min-w-[160px]"
            )

        def apply_bpso_config() -> None:
            state.workflow.config.n_particles = int(
                particles.value
            )
            state.workflow.config.n_iterations = int(
                iterations.value
            )
            state.workflow.config.inertia_weight = float(
                inertia.value
            )
            state.workflow.config.cognitive_coeff = float(
                c1.value
            )
            state.workflow.config.social_coeff = float(
                c2.value
            )
            state.workflow.config.feature_penalty_lambda = float(
                penalty.value
            )
            state.notify(
                "BPSO configuration updated."
            )

        with ui.row().classes(
            "gap-2 mt-4"
        ):
            ui.button(
                "Apply Configuration",
                icon="save",
                on_click=apply_bpso_config,
            ).props(
                "outline no-caps"
            )
            ui.button(
                "Run BPSO",
                icon="play_arrow",
                on_click=state.run_bpso,
            ).props(
                "unelevated no-caps"
            ).classes(
                "bg-blue-600 text-white"
            )

    live = state.progress or {}

    if state.running and state.running_stage == "BPSO optimization":
        with ui.card().classes(
            "w-full mt-4 p-5"
        ):
            ui.label("BPSO is running").classes(
                "text-lg font-semibold mb-3"
            )

            iteration = live.get("iteration", 0)
            total_iterations = live.get(
                "total_iterations",
                config.n_iterations,
            )
            best_fitness = live.get("best_fitness")
            selected_count = live.get(
                "selected_features"
            )

            with ui.row().classes(
                "w-full gap-4 flex-wrap"
            ):
                _metric_card(
                    "Iteration",
                    f"{iteration} / {total_iterations}",
                )
                _metric_card(
                    "Best Fitness",
                    _fmt(best_fitness, 6),
                )
                _metric_card(
                    "Selected",
                    selected_count if selected_count is not None else "—",
                )

            ui.linear_progress(
                min(
                    max(
                        _safe_float(iteration)
                        / max(
                            _safe_float(total_iterations),
                            1.0,
                        ),
                        0.0,
                    ),
                    1.0,
                )
            ).classes("w-full mt-3")

            fitness_details = live.get(
                "fitness_details",
                {},
            )
            if fitness_details:
                _key_value(
                    "Validation F1",
                    fitness_details.get("validation_f1"),
                )
                _key_value(
                    "Feature penalty",
                    fitness_details.get("feature_penalty"),
                )

    bpso_result = state.bpso_result

    if bpso_result:
        details = bpso_result.get(
            "best_details",
            {},
        )

        with ui.row().classes(
            "w-full mt-4 gap-4 flex-wrap"
        ):
            _metric_card(
                "Best Fitness",
                _fmt(
                    bpso_result.get(
                        "best_fitness"
                    ),
                    6,
                ),
            )
            _metric_card(
                "Validation F1",
                _fmt(
                    details.get(
                        "validation_f1"
                    ),
                    6,
                ),
            )
            _metric_card(
                "Selected Features",
                details.get(
                    "selected_features",
                    len(
                        state.workflow.selected_features
                    ),
                ),
            )
            _metric_card(
                "Feature Penalty",
                _fmt(
                    details.get(
                        "feature_penalty"
                    ),
                    6,
                ),
            )

        with ui.card().classes(
            "w-full mt-4 p-5"
        ):
            ui.label(
                "Research Fitness Function"
            ).classes(
                "text-lg font-semibold"
            )
            ui.label(
                "F(X) = Validation F1 − λ × (Selected Features / Total Features)"
            ).classes(
                "font-mono mt-2"
            )

            _key_value(
                "Validation F1",
                details.get("validation_f1"),
            )
            _key_value(
                "Selected Features",
                details.get("selected_features"),
            )
            _key_value(
                "Total Features",
                details.get("total_features"),
            )
            _key_value(
                "Feature Penalty",
                details.get("feature_penalty"),
            )
            _key_value(
                "Final Fitness",
                details.get("fitness"),
            )

        history = bpso_result.get(
            "history",
            [],
        )

        if history:
            chart_data = [
                {
                    "iteration": int(
                        item.get(
                            "iteration",
                            index + 1,
                        )
                    ),
                    "fitness": _safe_float(
                        item.get(
                            "best_fitness"
                        )
                    ),
                }
                for index, item in enumerate(
                    history
                )
            ]

            with ui.card().classes(
                "w-full mt-4 p-5"
            ):
                ui.label(
                    "BPSO Convergence"
                ).classes(
                    "text-lg font-semibold mb-3"
                )

                ui.echart(
                    {
                        "xAxis": {
                            "type": "category",
                            "data": [
                                x["iteration"]
                                for x in chart_data
                            ],
                        },
                        "yAxis": {
                            "type": "value"
                        },
                        "tooltip": {
                            "trigger": "axis"
                        },
                        "series": [
                            {
                                "name": "Best Fitness",
                                "type": "line",
                                "data": [
                                    x["fitness"]
                                    for x in chart_data
                                ],
                                "smooth": True,
                            }
                        ],
                    }
                ).classes(
                    "w-full h-80"
                )

        selected = state.workflow.selected_features

        with ui.card().classes(
            "w-full mt-4 p-5"
        ):
            ui.label(
                "Selected Features"
            ).classes(
                "text-lg font-semibold mb-3"
            )

            if selected:
                with ui.row().classes(
                    "gap-2 flex-wrap"
                ):
                    for feature in selected:
                        ui.chip(
                            feature,
                            icon="check",
                        )

                with ui.row().classes(
                    "gap-2 mt-4"
                ):
                    ui.button(
                        "Train Selected NN",
                        icon="model_training",
                        on_click=state.train_selected_model,
                    ).props(
                        "unelevated no-caps"
                    )
                    ui.button(
                        "Go to Comparison",
                        icon="compare_arrows",
                        on_click=lambda: state.change_page(
                            "comparison"
                        ),
                    ).props(
                        "outline no-caps"
                    )

        final_state = bpso_result.get(
            "last_state",
            {},
        )

        if final_state:
            with ui.card().classes(
                "w-full mt-4 p-5"
            ):
                ui.label(
                    "Final Particle State"
                ).classes(
                    "text-lg font-semibold mb-3"
                )

                _key_value(
                    "Iteration",
                    final_state.get("iteration"),
                )
                _key_value(
                    "Global Best Fitness",
                    final_state.get("gbest_fitness"),
                )
                _key_value(
                    "Global Best Selected Features",
                    final_state.get(
                        "selected_features"
                    ),
                )

                particles = final_state.get(
                    "particles",
                    [],
                )

                if particles:
                    rows = []
                    for particle in particles[:10]:
                        rows.append(
                            {
                                "particle": particle.get(
                                    "index"
                                ),
                                "fitness": _safe_float(
                                    particle.get(
                                        "fitness"
                                    )
                                ),
                                "selected": sum(
                                    particle.get(
                                        "position",
                                        [],
                                    )
                                ),
                                "position": "".join(
                                    str(x)
                                    for x in particle.get(
                                        "position",
                                        [],
                                    )[:60]
                                ),
                                "pbest": "".join(
                                    str(x)
                                    for x in particle.get(
                                        "pbest",
                                        [],
                                    )[:60]
                                ),
                            }
                        )

                    ui.table(
                        columns=[
                            {
                                "name": "particle",
                                "label": "Particle",
                                "field": "particle",
                            },
                            {
                                "name": "fitness",
                                "label": "Fitness",
                                "field": "fitness",
                            },
                            {
                                "name": "selected",
                                "label": "Selected",
                                "field": "selected",
                            },
                            {
                                "name": "position",
                                "label": "Position",
                                "field": "position",
                            },
                            {
                                "name": "pbest",
                                "label": "pBest",
                                "field": "pbest",
                            },
                        ],
                        rows=rows,
                    ).props(
                        "dense bordered wrap-cells"
                    ).classes(
                        "w-full"
                    )


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

def _render_neural_network(state: Any) -> None:
    _section_title(
        "Neural Network",
        "Controlled comparison: the same fixed one-hidden-layer model is trained with all features and with BPSO-selected features.",
    )

    config = state.workflow.config

    with ui.card().classes(
        "w-full p-5"
    ):
        ui.label(
            "Neural Network Configuration"
        ).classes(
            "text-lg font-semibold mb-3"
        )

        _key_value(
            "Hidden layers",
            1,
        )
        _key_value(
            "Hidden neurons",
            config.hidden_neurons,
        )
        _key_value(
            "Activation",
            "ReLU",
        )
        _key_value(
            "Optimizer",
            "Adam",
        )
        _key_value(
            "Learning rate",
            config.learning_rate,
        )
        _key_value(
            "Epochs",
            config.epochs,
        )
        _key_value(
            "Batch size",
            config.batch_size,
        )

        with ui.row().classes(
            "gap-2 mt-4"
        ):
            ui.button(
                "Train Baseline",
                icon="model_training",
                on_click=state.train_baseline,
            ).props(
                "unelevated no-caps"
            )
            ui.button(
                "Train BPSO Model",
                icon="model_training",
                on_click=state.train_selected_model,
            ).props(
                "outline no-caps"
            )

    baseline = getattr(
        state.workflow.pipeline,
        "baseline_metrics",
        None,
    )
    selected = getattr(
        state.workflow.pipeline,
        "selected_metrics",
        None,
    )

    if baseline or selected:
        with ui.row().classes(
            "w-full gap-4 mt-4 items-start"
        ):
            if baseline:
                with ui.card().classes(
                    "flex-1 p-5"
                ):
                    ui.label(
                        "Baseline Model"
                    ).classes(
                        "text-lg font-semibold"
                    )
                    test = baseline.get(
                        "test",
                        {},
                    )
                    _key_value(
                        "Accuracy",
                        test.get("accuracy"),
                    )
                    _key_value(
                        "Precision",
                        test.get("precision"),
                    )
                    _key_value(
                        "Recall",
                        test.get("recall"),
                    )
                    _key_value(
                        "F1",
                        test.get("f1"),
                    )
                    _key_value(
                        "Training time (s)",
                        test.get("training_seconds"),
                    )

            if selected:
                with ui.card().classes(
                    "flex-1 p-5"
                ):
                    ui.label(
                        "BPSO-Selected Model"
                    ).classes(
                        "text-lg font-semibold"
                    )
                    test = selected.get(
                        "test",
                        {},
                    )
                    _key_value(
                        "Accuracy",
                        test.get("accuracy"),
                    )
                    _key_value(
                        "Precision",
                        test.get("precision"),
                    )
                    _key_value(
                        "Recall",
                        test.get("recall"),
                    )
                    _key_value(
                        "F1",
                        test.get("f1"),
                    )
                    _key_value(
                        "Training time (s)",
                        test.get("training_seconds"),
                    )


# ---------------------------------------------------------------------------
# Fuzzy Risk
# ---------------------------------------------------------------------------

def _render_fuzzy(state: Any) -> None:
    _section_title(
        "Fuzzy Risk Assessment",
        "Three selected numerical inputs are mapped through Low / Medium / High membership functions and a configurable rule base.",
    )

    selected = state.workflow.selected_features

    if len(selected) < 3:
        ui.label(
            "At least three BPSO-selected features are required for the current fuzzy layer."
        ).classes(
            "text-sm text-gray-500"
        )
        return

    default_features = list(
        state.workflow.pipeline.fuzzy_features
        or selected[:3]
    )

    with ui.card().classes(
        "w-full p-5"
    ):
        ui.label(
            "Fuzzy Input Features"
        ).classes(
            "text-lg font-semibold mb-3"
        )

        feature_select = ui.select(
            selected,
            multiple=True,
            value=[
                f for f in default_features
                if f in selected
            ],
            label="Choose exactly three selected features",
        ).props(
            "use-chips"
        ).classes(
            "w-full"
        )

        def configure() -> None:
            chosen = list(
                feature_select.value
                or []
            )

            if len(chosen) != 3:
                state.notify(
                    "Select exactly three fuzzy features.",
                    type="negative",
                )
                return

            state.configure_fuzzy(
                chosen
            )

        ui.button(
            "Configure Fuzzy Layer",
            icon="settings",
            on_click=configure,
        ).props(
            "unelevated no-caps"
        ).classes(
            "mt-3"
        )

    fuzzy_engine = getattr(
        state.workflow.pipeline,
        "fuzzy_engine",
        None,
    )

    if fuzzy_engine is not None:
        with ui.card().classes(
            "w-full mt-4 p-5"
        ):
            ui.label(
                "Configured Inputs"
            ).classes(
                "text-lg font-semibold mb-3"
            )

            with ui.row().classes(
                "gap-2 flex-wrap"
            ):
                for name in state.workflow.pipeline.fuzzy_features:
                    ui.chip(
                        name,
                        icon="input",
                    )

            ui.label(
                "Membership labels"
            ).classes(
                "font-medium mt-4"
            )

            ui.label(
                "Low / Medium / High"
            ).classes(
                "text-sm text-gray-600"
            )

            ui.button(
                "Run Fuzzy Risk Assessment",
                icon="bolt",
                on_click=state.run_fuzzy,
            ).props(
                "unelevated no-caps"
            ).classes(
                "mt-4"
            )

    result = state.fuzzy_result

    if result:
        level = str(
            result.get(
                "risk_level",
                result.get(
                    "level",
                    "UNKNOWN",
                ),
            )
        ).upper()

        score = result.get(
            "risk_score",
            result.get(
                "score"
            ),
        )

        with ui.row().classes(
            "w-full gap-4 mt-4"
        ):
            _metric_card(
                "Risk Level",
                level,
            )
            _metric_card(
                "Risk Score",
                _fmt(score, 4),
            )

        with ui.card().classes(
            "w-full mt-4 p-5"
        ):
            ui.label(
                "Fuzzy Interpretation"
            ).classes(
                "text-lg font-semibold mb-3"
            )

            memberships = result.get(
                "memberships",
                result.get(
                    "aggregated",
                    {},
                ),
            )

            if isinstance(
                memberships,
                dict,
            ):
                for key, value in memberships.items():
                    _key_value(
                        key,
                        value,
                    )

            explanation = result.get(
                "explanation"
            )

            if explanation:
                ui.label(
                    str(explanation)
                ).classes(
                    "text-sm text-gray-600 mt-3"
                )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _render_comparison(state: Any) -> None:
    """
    Value-added comparison for the proposed BPSO approach.

    Controlled experiment:
        Baseline = all usable preprocessed features + fixed NN
        Proposed = BPSO-selected features + same fixed NN

    Both models are evaluated on the same untouched test split.

    This page answers:
        1. What did BPSO save?
        2. What did BPSO gain/retain?
        3. What was the performance trade-off?
    """
    _section_title(
        "Comparison — What Value Does BPSO Add?",
        "Controlled experiment: same dataset, same preprocessing, same neural-network architecture, same test set — only the feature set changes.",
    )

    prepared = getattr(
        state.workflow.pipeline,
        "prepared",
        None,
    )
    comparison = getattr(
        state.workflow.pipeline,
        "comparison",
        None,
    )

    # ------------------------------------------------------------------
    # Comparison setup
    # ------------------------------------------------------------------
    with ui.card().classes("w-full p-5"):
        ui.label("Experimental Design").classes(
            "text-lg font-semibold"
        )

        with ui.column().classes("gap-1 mt-3"):
            ui.label(
                "Baseline: all usable preprocessed features → fixed neural network → prediction"
            ).classes("text-sm")
            ui.label(
                "Proposed: BPSO-selected features → the same fixed neural network → prediction"
            ).classes("text-sm")
            ui.label(
                "Both models use the same train / validation / test split. This isolates the effect of feature selection."
            ).classes("text-xs text-gray-500 mt-1")

        if prepared is None:
            ui.label(
                "Load and prepare the dataset first."
            ).classes("text-sm text-gray-500 mt-4")
            return

        original_count = len(prepared.feature_names)
        selected_count = len(state.workflow.selected_features)

        with ui.row().classes(
            "w-full gap-3 mt-4 flex-wrap"
        ):
            _metric_card(
                "Baseline Input",
                original_count,
                "all usable features",
            )
            _metric_card(
                "Proposed Input",
                selected_count if selected_count else "Pending",
                "BPSO-selected features",
            )
            _metric_card(
                "Features Removed",
                (
                    original_count - selected_count
                    if selected_count
                    else "Pending"
                ),
                "removed from model input",
            )
            _metric_card(
                "Feature Reduction",
                (
                    f"{100.0 * (1.0 - selected_count / original_count):.2f}%"
                    if selected_count and original_count
                    else "Pending"
                ),
                "after BPSO",
            )

        if not state.workflow.selected_features:
            ui.label(
                "BPSO has not produced a selected feature subset yet. Run BPSO before generating the final comparison."
            ).classes("text-sm text-amber-700 mt-4")

        ui.button(
            "Run / Refresh Comparison",
            icon="compare_arrows",
            on_click=state.evaluate,
            disable=state.running,
        ).props(
            "unelevated no-caps"
        ).classes(
            "mt-4 bg-blue-600 text-white"
        )

    if comparison is None:
        with ui.card().classes("w-full mt-4 p-5"):
            ui.label(
                "Comparison not generated yet"
            ).classes("text-lg font-semibold")
            ui.label(
                "Run the comparison to train/evaluate the baseline and BPSO-selected models and calculate the value added by feature selection."
            ).classes("text-sm text-gray-500 mt-2")
        return

    baseline = comparison.baseline
    bpso = comparison.bpso

    baseline_feature_count = int(comparison.original_features)
    selected_feature_count = int(comparison.selected_features)
    removed_feature_count = max(
        baseline_feature_count - selected_feature_count,
        0,
    )
    reduction_percent = float(
        comparison.reduction_percent
    )

    baseline_training = _safe_float(
        getattr(baseline, "training_seconds", 0.0)
    )
    bpso_training = _safe_float(
        getattr(bpso, "training_seconds", 0.0)
    )
    baseline_inference = _safe_float(
        getattr(baseline, "inference_seconds", 0.0)
    )
    bpso_inference = _safe_float(
        getattr(bpso, "inference_seconds", 0.0)
    )

    training_saved = baseline_training - bpso_training
    inference_saved = baseline_inference - bpso_inference

    def percent_change(
        before: float,
        after: float,
    ) -> float | None:
        if abs(before) < 1e-12:
            return None
        return 100.0 * (before - after) / before

    training_saving_percent = percent_change(
        baseline_training,
        bpso_training,
    )
    inference_saving_percent = percent_change(
        baseline_inference,
        bpso_inference,
    )

    f1_difference = float(
        bpso.f1 - baseline.f1
    )
    accuracy_difference = float(
        bpso.accuracy - baseline.accuracy
    )
    precision_difference = float(
        bpso.precision - baseline.precision
    )
    recall_difference = float(
        bpso.recall - baseline.recall
    )

    f1_retention = None
    if abs(float(baseline.f1)) > 1e-12:
        f1_retention = (
            float(bpso.f1)
            / float(baseline.f1)
            * 100.0
        )

    hidden_neurons = int(
        getattr(
            state.workflow.config,
            "hidden_neurons",
            32,
        )
    )
    baseline_input_connections = (
        baseline_feature_count * hidden_neurons
    )
    bpso_input_connections = (
        selected_feature_count * hidden_neurons
    )
    input_connections_saved = max(
        baseline_input_connections
        - bpso_input_connections,
        0,
    )

    # ------------------------------------------------------------------
    # Executive result — the first thing an examiner should see
    # ------------------------------------------------------------------
    with ui.card().classes("w-full mt-4 p-5"):
        ui.label(
            "BPSO Value Added"
        ).classes("text-xl font-bold")
        ui.label(
            "The following values are calculated from the current experiment; they are not placeholder research results."
        ).classes("text-xs text-gray-500 mt-1 mb-4")

        with ui.row().classes(
            "w-full gap-4 flex-wrap"
        ):
            _metric_card(
                "Feature Reduction",
                f"{reduction_percent:.2f}%",
                f"{removed_feature_count} fewer features",
            )
            _metric_card(
                "F1 Retention",
                (
                    f"{f1_retention:.2f}%"
                    if f1_retention is not None
                    else "N/A"
                ),
                "BPSO F1 / baseline F1",
            )
            _metric_card(
                "Training Time Change",
                (
                    f"{training_saving_percent:+.2f}%"
                    if training_saving_percent is not None
                    else "N/A"
                ),
                "positive = time saved",
            )
            _metric_card(
                "Inference Time Change",
                (
                    f"{inference_saving_percent:+.2f}%"
                    if inference_saving_percent is not None
                    else "N/A"
                ),
                "positive = time saved",
            )

    # ------------------------------------------------------------------
    # What did we save?
    # ------------------------------------------------------------------
    with ui.card().classes("w-full mt-4 p-5"):
        ui.label(
            "What Did BPSO Save?"
        ).classes("text-xl font-bold")
        ui.label(
            "BPSO reduces the dimensionality of the model input. The time values below come from the actual evaluation objects."
        ).classes("text-sm text-gray-500 mt-1 mb-4")

        savings_rows = [
            {
                "resource": "Features",
                "baseline": baseline_feature_count,
                "bpso": selected_feature_count,
                "saved": removed_feature_count,
                "change": reduction_percent,
            },
            {
                "resource": "Training time (s)",
                "baseline": baseline_training,
                "bpso": bpso_training,
                "saved": training_saved,
                "change": training_saving_percent,
            },
            {
                "resource": "Inference time (s)",
                "baseline": baseline_inference,
                "bpso": bpso_inference,
                "saved": inference_saved,
                "change": inference_saving_percent,
            },
            {
                "resource": "Input-layer connections",
                "baseline": baseline_input_connections,
                "bpso": bpso_input_connections,
                "saved": input_connections_saved,
                "change": (
                    100.0 * input_connections_saved
                    / baseline_input_connections
                    if baseline_input_connections
                    else None
                ),
            },
        ]

        ui.table(
            columns=[
                {
                    "name": "resource",
                    "label": "Resource",
                    "field": "resource",
                    "align": "left",
                },
                {
                    "name": "baseline",
                    "label": "Baseline",
                    "field": "baseline",
                    "format": lambda value: (
                        f"{float(value):.4f}"
                        if isinstance(value, float)
                        else f"{value:,}"
                    ),
                },
                {
                    "name": "bpso",
                    "label": "BPSO",
                    "field": "bpso",
                    "format": lambda value: (
                        f"{float(value):.4f}"
                        if isinstance(value, float)
                        else f"{value:,}"
                    ),
                },
                {
                    "name": "saved",
                    "label": "Saved",
                    "field": "saved",
                    "format": lambda value: (
                        "N/A"
                        if value is None
                        else (
                            f"{float(value):+.4f}"
                            if isinstance(value, float)
                            else f"{value:,}"
                        )
                    ),
                },
                {
                    "name": "change",
                    "label": "% Reduction",
                    "field": "change",
                    "format": lambda value: (
                        "N/A"
                        if value is None
                        else f"{float(value):+.2f}%"
                    ),
                },
            ],
            rows=savings_rows,
        ).props(
            "dense bordered"
        ).classes("w-full")

    # ------------------------------------------------------------------
    # What did we gain / retain?
    # ------------------------------------------------------------------
    with ui.card().classes("w-full mt-4 p-5"):
        ui.label(
            "What Did We Gain or Retain?"
        ).classes("text-xl font-bold")
        ui.label(
            "Performance must be interpreted together with feature reduction. A positive metric delta means the BPSO model improved that metric; a negative delta means a trade-off occurred."
        ).classes("text-sm text-gray-500 mt-1 mb-4")

        performance_rows = [
            {
                "metric": "Accuracy",
                "baseline": float(baseline.accuracy),
                "bpso": float(bpso.accuracy),
                "difference": accuracy_difference,
            },
            {
                "metric": "Precision",
                "baseline": float(baseline.precision),
                "bpso": float(bpso.precision),
                "difference": precision_difference,
            },
            {
                "metric": "Recall",
                "baseline": float(baseline.recall),
                "bpso": float(bpso.recall),
                "difference": recall_difference,
            },
            {
                "metric": "F1",
                "baseline": float(baseline.f1),
                "bpso": float(bpso.f1),
                "difference": f1_difference,
            },
        ]

        ui.table(
            columns=[
                {
                    "name": "metric",
                    "label": "Metric",
                    "field": "metric",
                    "align": "left",
                },
                {
                    "name": "baseline",
                    "label": "All Features + NN",
                    "field": "baseline",
                    "format": lambda value: f"{float(value):.4f}",
                },
                {
                    "name": "bpso",
                    "label": "BPSO Features + NN",
                    "field": "bpso",
                    "format": lambda value: f"{float(value):.4f}",
                },
                {
                    "name": "difference",
                    "label": "Δ (BPSO − Baseline)",
                    "field": "difference",
                    "format": lambda value: f"{float(value):+.4f}",
                },
            ],
            rows=performance_rows,
        ).props(
            "dense bordered"
        ).classes("w-full")

        ui.echart(
            {
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {"type": "shadow"},
                },
                "legend": {
                    "data": [
                        "All Features + NN",
                        "BPSO Features + NN",
                    ]
                },
                "xAxis": {
                    "type": "category",
                    "data": [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                    ],
                },
                "yAxis": {
                    "type": "value",
                    "min": 0,
                    "max": 1,
                },
                "series": [
                    {
                        "name": "All Features + NN",
                        "type": "bar",
                        "data": [
                            float(baseline.accuracy),
                            float(baseline.precision),
                            float(baseline.recall),
                            float(baseline.f1),
                        ],
                    },
                    {
                        "name": "BPSO Features + NN",
                        "type": "bar",
                        "data": [
                            float(bpso.accuracy),
                            float(bpso.precision),
                            float(bpso.recall),
                            float(bpso.f1),
                        ],
                    },
                ],
            }
        ).classes("w-full h-80 mt-4")

    # ------------------------------------------------------------------
    # Trade-off assessment
    # ------------------------------------------------------------------
    if f1_difference > 1e-12:
        f1_assessment = (
            "BPSO improved F1 while using fewer features."
        )
        f1_icon = "trending_up"
    elif abs(f1_difference) <= 1e-12:
        f1_assessment = (
            "BPSO retained the same F1 while using fewer features."
        )
        f1_icon = "horizontal_rule"
    else:
        f1_assessment = (
            "BPSO reduced features but introduced a lower F1; this is the measured performance trade-off."
        )
        f1_icon = "trending_down"

    if training_saved > 0:
        training_assessment = (
            f"Training time decreased by {training_saved:.4f} seconds."
        )
        training_icon = "speed"
    elif training_saved < 0:
        training_assessment = (
            f"Training time increased by {abs(training_saved):.4f} seconds in this run."
        )
        training_icon = "schedule"
    else:
        training_assessment = "Training time was unchanged in this run."
        training_icon = "horizontal_rule"

    if inference_saved > 0:
        inference_assessment = (
            f"Inference time decreased by {inference_saved:.6f} seconds."
        )
        inference_icon = "speed"
    elif inference_saved < 0:
        inference_assessment = (
            f"Inference time increased by {abs(inference_saved):.6f} seconds in this run."
        )
        inference_icon = "schedule"
    else:
        inference_assessment = "Inference time was unchanged in this run."
        inference_icon = "horizontal_rule"

    with ui.card().classes("w-full mt-4 p-5"):
        ui.label(
            "Trade-off and Research Interpretation"
        ).classes("text-xl font-bold")

        with ui.column().classes("gap-3 mt-4"):
            with ui.row().classes("items-start gap-3"):
                ui.icon(f1_icon).classes("text-2xl")
                ui.label(f1_assessment).classes("text-sm leading-6")

            with ui.row().classes("items-start gap-3"):
                ui.icon(training_icon).classes("text-2xl")
                ui.label(training_assessment).classes("text-sm leading-6")

            with ui.row().classes("items-start gap-3"):
                ui.icon(inference_icon).classes("text-2xl")
                ui.label(inference_assessment).classes("text-sm leading-6")

            with ui.row().classes("items-start gap-3"):
                ui.icon("memory").classes("text-2xl")
                ui.label(
                    f"The first hidden-layer input connections fall from {baseline_input_connections:,} to {bpso_input_connections:,}, a reduction of {input_connections_saved:,} connections ({100.0 * input_connections_saved / baseline_input_connections:.2f}%)."
                    if baseline_input_connections
                    else "Input-layer connection change is unavailable."
                ).classes("text-sm leading-6")

    # ------------------------------------------------------------------
    # Final value statement
    # ------------------------------------------------------------------
    if (
        f1_difference >= -0.01
        and reduction_percent > 0
    ):
        overall = (
            "BPSO demonstrates a feature-efficiency benefit in this experiment: "
            f"it removes {removed_feature_count} of {baseline_feature_count} features "
            f"({reduction_percent:.2f}% reduction) while retaining "
            f"{f1_retention:.2f}% of baseline F1."
            if f1_retention is not None
            else (
                "BPSO demonstrates a feature-efficiency benefit in this experiment by reducing the model input dimension."
            )
        )
    else:
        overall = (
            "BPSO reduced the feature space, but the measured classification trade-off should be considered when interpreting the result."
        )

    with ui.card().classes(
        "w-full mt-4 p-6 border-2"
    ):
        ui.label(
            "Final Value Added by BPSO"
        ).classes("text-xl font-bold")
        ui.label(
            overall
        ).classes("text-base mt-3 leading-7")

        ui.label(
            "This conclusion is generated from the current experiment; it should be recomputed whenever the dataset, split, or model settings change."
        ).classes("text-xs text-gray-500 mt-3")

    # ------------------------------------------------------------------
    # Feature list
    # ------------------------------------------------------------------
    with ui.row().classes(
        "w-full gap-4 mt-4 items-start"
    ):
        with ui.card().classes("flex-1 p-5"):
            ui.label(
                "Selected Features"
            ).classes("text-lg font-semibold mb-3")

            selected_features = list(
                state.workflow.selected_features
            )

            if selected_features:
                with ui.row().classes(
                    "gap-2 flex-wrap"
                ):
                    for feature in selected_features:
                        ui.chip(
                            feature,
                            icon="check",
                        )
            else:
                ui.label(
                    "Selected feature list is unavailable."
                ).classes("text-sm text-gray-500")

        with ui.card().classes("flex-1 p-5"):
            ui.label(
                "Model Input Interpretation"
            ).classes("text-lg font-semibold mb-3")
            _key_value(
                "Baseline input size",
                baseline_feature_count,
            )
            _key_value(
                "Proposed input size",
                selected_feature_count,
            )
            _key_value(
                "Remaining feature fraction",
                f"{100.0 * selected_feature_count / baseline_feature_count:.2f}%"
                if baseline_feature_count
                else "N/A",
            )
            _key_value(
                "F1 retention",
                f"{f1_retention:.2f}%"
                if f1_retention is not None
                else "N/A",
            )


# ---------------------------------------------------------------------------
# Single record
# ---------------------------------------------------------------------------

def _render_single_record(state: Any) -> None:
    _section_title(
        "Single Traffic Analysis",
        "Analyze one record through the trained BPSO-selected model and the fuzzy risk layer.",
    )

    raw_df = getattr(
        state.workflow.pipeline,
        "raw_dataframe",
        None,
    )

    if raw_df is None:
        ui.label(
            "Load a dataset first."
        ).classes(
            "text-sm text-gray-500"
        )
        return

    prepared = getattr(
        state.workflow.pipeline,
        "prepared",
        None,
    )

    if prepared is None:
        ui.label(
            "Prepare the dataset before analyzing a single record."
        ).classes(
            "text-sm text-gray-500"
        )
        return

    with ui.card().classes(
        "w-full p-5"
    ):
        max_index = max(
            len(raw_df) - 1,
            0,
        )

        row_index = ui.number(
            "Row index",
            value=0,
            min=0,
            max=max_index,
            step=1,
        ).classes(
            "w-full"
        )

        ui.button(
            "Analyze Record",
            icon="analytics",
            on_click=lambda: state.predict_record(
                int(row_index.value or 0)
            ),
        ).props(
            "unelevated no-caps"
        ).classes(
            "mt-3"
        )

    result = state.prediction_result

    if result:
        with ui.row().classes(
            "w-full mt-4 gap-4 flex-wrap"
        ):
            _metric_card(
                "Prediction",
                result.get(
                    "prediction",
                    "—",
                ),
            )
            _metric_card(
                "Confidence",
                _fmt(
                    result.get(
                        "confidence"
                    ),
                    4,
                ),
            )

            fuzzy = result.get(
                "fuzzy"
            ) or result.get(
                "fuzzy_result"
            ) or {}

            _metric_card(
                "Risk Level",
                fuzzy.get(
                    "risk_level",
                    fuzzy.get(
                        "level",
                        "—",
                    ),
                ),
            )
            _metric_card(
                "Risk Score",
                _fmt(
                    fuzzy.get(
                        "risk_score"
                    ),
                    4,
                ),
            )

        with ui.card().classes(
            "w-full mt-4 p-5"
        ):
            ui.label(
                "Selected Features Used"
            ).classes(
                "text-lg font-semibold mb-3"
            )

            for feature in result.get(
                "selected_features",
                state.workflow.selected_features,
            ):
                ui.chip(
                    feature,
                    icon="check",
                )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _render_report(state: Any) -> None:
    _section_title(
        "Experiment Report",
        "Persisted artifacts and a compact report for presentation, inspection, and reuse.",
    )

    artifact_dir = Path(
        state.workflow.config.artifacts_dir
    )

    with ui.card().classes(
        "w-full p-5"
    ):
        _key_value(
            "Artifacts directory",
            artifact_dir,
        )

        files = [
            "results.json",
            "bpso_history.csv",
            "selected_features.json",
            "fuzzy_config.json",
            "preprocessor.joblib",
            "baseline_model.joblib",
            "selected_model.joblib",
        ]

        for filename in files:
            path = artifact_dir / filename

            with ui.row().classes(
                "w-full justify-between items-center py-1"
            ):
                ui.label(filename)
                ui.badge(
                    "Available"
                    if path.exists()
                    else "Not generated",
                    color=(
                        "positive"
                        if path.exists()
                        else "grey"
                    ),
                )

        with ui.row().classes(
            "gap-2 mt-4"
        ):
            ui.button(
                "Export Report",
                icon="download",
                on_click=state.export_report,
            ).props(
                "unelevated no-caps"
            )
            ui.button(
                "Run Complete Experiment",
                icon="play_arrow",
                on_click=state.run_all,
            ).props(
                "outline no-caps"
            )

    payload = state.result_payload

    if payload:
        with ui.card().classes(
            "w-full mt-4 p-5"
        ):
            ui.label(
                "Current Result Payload"
            ).classes(
                "text-lg font-semibold mb-3"
            )

            preview = json.dumps(
                payload,
                indent=2,
                default=str,
            )

            ui.code(
                preview[:12000],
                language="json",
            ).classes(
                "w-full max-h-[500px] overflow-auto"
            )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGE_RENDERERS = {
    "dashboard": _render_dashboard,
    "dataset": _render_dataset,
    "preprocessing": _render_preprocessing,
    "bpso": _render_bpso,
    "neural_network": _render_neural_network,
    "fuzzy": _render_fuzzy,
    "comparison": _render_comparison,
    "single_record": _render_single_record,
    "report": _render_report,
}


def render_workspace(
    state: Any,
) -> None:
    """Render the active application page."""

    renderer = PAGE_RENDERERS.get(
        state.current_page,
        _render_dashboard,
    )

    renderer(state)
