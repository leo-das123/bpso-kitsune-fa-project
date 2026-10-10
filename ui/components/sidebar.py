from __future__ import annotations

from collections.abc import Callable

from nicegui import ui


NAV_ITEMS: list[tuple[str, str, str]] = [
    ("dashboard", "Dashboard", "dashboard"),
    ("dataset", "Dataset", "dataset"),
    ("preprocessing", "Preprocessing", "tune"),
    ("bpso", "BPSO Feature Selection", "hub"),
    ("neural_network", "Neural Network", "account_tree"),
    ("fuzzy", "Fuzzy Risk Assessment", "psychology"),
    ("comparison", "Comparison", "compare_arrows"),
    ("single_record", "Single Traffic Analysis", "analytics"),
    ("report", "Experiment Report", "description"),
]


def render_sidebar(
    current_page: str,
    on_change: Callable[[str], None],
    stage_status: dict[str, str] | None = None,
    on_run_all: Callable[[], None] | None = None,
) -> None:
    """Render the research navigation sidebar."""

    stage_status = stage_status or {}

    with ui.column().classes("w-full h-full p-4 gap-1"):
        with ui.row().classes("w-full items-center gap-3 mb-1"):
            ui.icon("security").classes("text-blue-600 text-2xl")
            with ui.column().classes("gap-0"):
                ui.label("BPSO-IntrusionGuard").classes(
                    "text-lg font-bold leading-5"
                )
                ui.label("Research Workspace").classes(
                    "text-xs text-gray-500"
                )

        ui.separator().classes("my-3")

        for key, label, icon in NAV_ITEMS:
            is_active = key == current_page

            if key == "dashboard":
                status_icon = "home"
            else:
                backend_stage = {
                    "dataset": "dataset",
                    "preprocessing": "preprocessing",
                    "bpso": "bpso",
                    "neural_network": "selected_model",
                    "fuzzy": "fuzzy",
                    "comparison": "evaluation",
                }.get(key)

                status = (
                    stage_status.get(backend_stage, "pending")
                    if backend_stage
                    else "pending"
                )
                status_icon = (
                    "check_circle"
                    if status == "complete"
                    else icon
                )

            button = ui.button(
                label,
                icon=status_icon,
                on_click=lambda _event, page=key: on_change(page),
            ).props("flat no-caps align=left")

            classes = "w-full justify-start text-left"
            if is_active:
                classes += " bg-blue-50 text-blue-700 font-semibold"

            button.classes(classes)

        ui.space()
        ui.separator().classes("my-3")

        if on_run_all is not None:
            ui.button(
                "Run Complete Experiment",
                icon="play_arrow",
                on_click=on_run_all,
            ).props("unelevated no-caps").classes(
                "w-full bg-blue-600 text-white"
            )

        with ui.card().classes("w-full p-3 mt-2 bg-gray-50"):
            ui.label("Research Flow").classes(
                "text-xs font-semibold text-gray-700"
            )
            ui.label(
                "CSV → Preprocessing → BPSO → NN → Fuzzy → Evaluation"
            ).classes(
                "text-xs text-gray-500 leading-4 mt-1"
            )

        ui.label("BPSO-IntrusionGuard").classes(
            "text-xs text-gray-400 mt-1"
        )
        ui.label("Educational / research prototype").classes(
            "text-[11px] text-gray-400"
        )
