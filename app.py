from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from nicegui import ui

from bpso_ids.config import make_config
from workflows.experiment import ExperimentWorkflow
from ui.components.sidebar import render_sidebar
from ui.pages.workspace import render_workspace


APP_TITLE = "BPSO-IntrusionGuard"
HOST = "0.0.0.0"
PORT = 8080

# Resolve the upload directory from the project root so validation and
# saving use exactly the same absolute filesystem location.
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AppState:
    """
    Per-page state for the NiceGUI application.

    Background workers never call UI methods directly. They only update
    this state. A small client-side NiceGUI timer observes the state and
    performs all UI notifications/refreshes inside the active client context.
    """

    current_page: str = "dashboard"

    dataset_path: str = ""
    pending_dataset_path: str = ""
    target_column: str | None = None
    drop_columns: list[str] = field(default_factory=list)

    preview_df: pd.DataFrame | None = None
    preview_columns: list[str] = field(default_factory=list)
    preview_error: str = ""

    workflow: ExperimentWorkflow = field(
        default_factory=lambda: ExperimentWorkflow(make_config("demo"))
    )

    bpso_result: dict[str, Any] | None = None
    fuzzy_result: dict[str, Any] | None = None
    prediction_result: dict[str, Any] | None = None
    result_payload: dict[str, Any] | None = None

    progress: dict[str, Any] = field(default_factory=dict)

    running: bool = False
    running_stage: str = ""
    state_version: int = 0
    rendered_version: int = 0

    pending_notice: tuple[str, str] | None = None

    refresh_main: Any = None
    refresh_sidebar: Any = None
    refresh_shell: Any = None

    def attach_progress_callback(self) -> None:
        self.workflow.set_progress_callback(self._on_progress)

    def _touch(self) -> None:
        self.state_version += 1

    def _on_progress(self, event: dict[str, Any]) -> None:
        # This callback may run from the BPSO worker thread.
        # Do not call any NiceGUI UI method here.
        self.progress = dict(event)
        self._touch()

    def queue_notice(
        self,
        message: str,
        kind: str = "positive",
    ) -> None:
        self.pending_notice = (message, kind)
        self._touch()

    def drain_notice(self) -> tuple[str, str] | None:
        notice = self.pending_notice
        self.pending_notice = None
        return notice

    def change_page(self, page: str) -> None:
        self.current_page = page
        self._touch()

    def refresh(self) -> None:
        if self.refresh_main is not None:
            self.refresh_main()

        if self.refresh_sidebar is not None:
            self.refresh_sidebar()

        if self.refresh_shell is not None:
            self.refresh_shell()

        self.rendered_version = self.state_version

    def set_target(self, target: str | None) -> None:
        self.target_column = str(target) if target else None
        self._touch()

    def set_drop_columns(self, columns: list[str] | None) -> None:
        self.drop_columns = list(columns or [])
        self._touch()

    # ------------------------------------------------------------------
    # Dataset upload / preview / load
    # ------------------------------------------------------------------

    async def handle_upload(self, event: Any) -> None:
        """
        Handle a NiceGUI 3.x upload event.

        NiceGUI's current FileUpload.save() API is asynchronous, so the
        upload handler must await it before attempting to preview/validate
        the saved CSV.
        """
        try:
            uploaded_file = event.file
            filename = Path(
                str(uploaded_file.name)
            ).name

            if not filename:
                self.queue_notice(
                    "The uploaded file has no valid filename.",
                    "negative",
                )
                return

            if not filename.lower().endswith(".csv"):
                self.queue_notice(
                    "Please upload a CSV file.",
                    "negative",
                )
                return

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            destination = (
                UPLOAD_DIR
                / f"{timestamp}_{filename}"
            ).resolve()

            # IMPORTANT: FileUpload.save() is async in current NiceGUI.
            # The previous version called it without await, so the coroutine
            # was never completed and the subsequent existence check failed.
            await uploaded_file.save(
                destination
            )

            if not destination.is_file():
                raise IOError(
                    f"Uploaded file was not saved to: {destination}"
                )

            self.pending_dataset_path = str(
                destination
            )
            self.dataset_path = ""
            self.preview_error = ""

            # Validate and preview only. Uploading never starts BPSO.
            self.preview_dataset(
                str(destination)
            )

            self.queue_notice(
                "CSV uploaded. Review the preview and choose the target column before loading it."
            )

        except Exception as exc:
            self.queue_notice(
                f"Upload failed: {exc}",
                "negative",
            )

        self._touch()

    def preview_dataset(self, path: str) -> None:
        csv_path = Path(str(path).strip())

        if not csv_path.exists():
            self.preview_df = None
            self.preview_columns = []
            self.preview_error = f"CSV not found: {csv_path}"
            self._touch()
            return

        if csv_path.suffix.lower() != ".csv":
            self.preview_df = None
            self.preview_columns = []
            self.preview_error = "Selected file is not a CSV."
            self._touch()
            return

        try:
            preview = pd.read_csv(csv_path, nrows=10)

            if preview.empty or len(preview.columns) < 2:
                raise ValueError(
                    "The CSV must contain at least two columns."
                )

            self.preview_df = preview
            self.preview_columns = [
                str(column) for column in preview.columns
            ]
            self.preview_error = ""
            self.pending_dataset_path = str(csv_path)

            if self.target_column not in self.preview_columns:
                self.target_column = None

        except Exception as exc:
            self.preview_df = None
            self.preview_columns = []
            self.preview_error = str(exc)

        self._touch()

    def load_existing_for_preview(self, path: str) -> None:
        self.preview_dataset(path)
        if not self.preview_error:
            self.queue_notice(
                "CSV validated. Choose the target and click Load Dataset."
            )

    def load_dataset(self) -> None:
        path = self.pending_dataset_path.strip()

        if not path:
            self.queue_notice(
                "Upload a CSV or enter an existing CSV path first.",
                "warning",
            )
            return

        try:
            profile = self.workflow.load_csv(
                path,
                target_column=self.target_column,
                drop_columns=self.drop_columns,
            )

            self.dataset_path = path

            detected = profile.get("target_column")
            candidates = profile.get(
                "possible_target_columns",
                [],
            )

            if self.target_column is None:
                self.target_column = (
                    detected
                    or (
                        candidates[0]
                        if candidates
                        else None
                    )
                )

            self.queue_notice(
                f"Dataset loaded: {Path(path).name}"
            )
            self.current_page = "dataset"

        except Exception as exc:
            self.queue_notice(
                f"Dataset load failed: {exc}",
                "negative",
            )

        self._touch()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def prepare_dataset(self) -> None:
        if self.workflow.dataset_profile is None:
            self.queue_notice(
                "Load the dataset first.",
                "warning",
            )
            return

        if not self.target_column:
            self.queue_notice(
                "Select a target column first.",
                "warning",
            )
            return

        if self.running:
            return

        self.running = True
        self.running_stage = "Preprocessing"
        self._touch()

        asyncio.create_task(
            self._prepare_dataset_worker(
                self.target_column,
                list(self.drop_columns),
            )
        )

    async def _prepare_dataset_worker(
        self,
        target: str,
        drop_columns: list[str],
    ) -> None:
        try:
            await asyncio.to_thread(
                self.workflow.prepare_dataset,
                target,
                drop_columns,
            )
            self.queue_notice(
                "Preprocessing completed."
            )
        except Exception as exc:
            self.queue_notice(
                f"Preprocessing failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    # ------------------------------------------------------------------
    # BPSO
    # ------------------------------------------------------------------

    def run_bpso(self) -> None:
        if self.workflow.pipeline.prepared is None:
            self.queue_notice(
                "Prepare the dataset before running BPSO.",
                "warning",
            )
            return

        if self.running:
            return

        self.running = True
        self.running_stage = "BPSO optimization"
        self._touch()

        asyncio.create_task(self._run_bpso_worker())

    async def _run_bpso_worker(self) -> None:
        try:
            await asyncio.to_thread(
                self.workflow.initialize_bpso
            )
            result = await asyncio.to_thread(
                self.workflow.run_bpso
            )
            self.bpso_result = result
            self.queue_notice(
                "BPSO feature selection completed."
            )
        except Exception as exc:
            self.queue_notice(
                f"BPSO failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    # ------------------------------------------------------------------
    # Neural network
    # ------------------------------------------------------------------

    def train_baseline(self) -> None:
        if self.workflow.pipeline.prepared is None:
            self.queue_notice(
                "Prepare the dataset first.",
                "warning",
            )
            return

        if self.running:
            return

        self.running = True
        self.running_stage = "Baseline neural network"
        self._touch()

        asyncio.create_task(
            self._train_baseline_worker()
        )

    async def _train_baseline_worker(self) -> None:
        try:
            await asyncio.to_thread(
                self.workflow.train_baseline
            )
            self.queue_notice(
                "Baseline neural network trained."
            )
        except Exception as exc:
            self.queue_notice(
                f"Baseline training failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    def train_selected_model(self) -> None:
        if self.workflow.pipeline.prepared is None:
            self.queue_notice(
                "Prepare the dataset first.",
                "warning",
            )
            return

        if not self.workflow.selected_features:
            self.queue_notice(
                "Run BPSO before training the selected model.",
                "warning",
            )
            return

        if self.running:
            return

        self.running = True
        self.running_stage = "Selected-feature neural network"
        self._touch()

        asyncio.create_task(
            self._train_selected_worker()
        )

    async def _train_selected_worker(self) -> None:
        try:
            await asyncio.to_thread(
                self.workflow.train_selected_model
            )
            self.queue_notice(
                "Selected-feature neural network trained."
            )
        except Exception as exc:
            self.queue_notice(
                f"Selected-model training failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    # ------------------------------------------------------------------
    # Fuzzy
    # ------------------------------------------------------------------

    def configure_fuzzy(
        self,
        features: list[str] | None = None,
    ) -> None:
        if self.running:
            return

        self.running = True
        self.running_stage = "Fuzzy configuration"
        self._touch()

        asyncio.create_task(
            self._configure_fuzzy_worker(features)
        )

    async def _configure_fuzzy_worker(
        self,
        features: list[str] | None,
    ) -> None:
        try:
            await asyncio.to_thread(
                self.workflow.configure_fuzzy,
                features,
            )
            self.queue_notice(
                "Fuzzy layer configured."
            )
        except Exception as exc:
            self.queue_notice(
                f"Fuzzy configuration failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    def run_fuzzy(self) -> None:
        if self.running:
            return

        self.running = True
        self.running_stage = "Fuzzy risk assessment"
        self._touch()

        asyncio.create_task(
            self._run_fuzzy_worker()
        )

    async def _run_fuzzy_worker(self) -> None:
        try:
            self.fuzzy_result = await asyncio.to_thread(
                self.workflow.run_fuzzy
            )
            self.queue_notice(
                "Fuzzy risk assessment completed."
            )
        except Exception as exc:
            self.queue_notice(
                f"Fuzzy assessment failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> None:
        if self.running:
            return

        self.running = True
        self.running_stage = "Evaluation"
        self._touch()

        asyncio.create_task(
            self._evaluate_worker()
        )

    async def _evaluate_worker(self) -> None:
        try:
            await asyncio.to_thread(
                self.workflow.evaluate
            )
            self.queue_notice(
                "Model comparison updated."
            )
        except Exception as exc:
            self.queue_notice(
                f"Evaluation failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    # ------------------------------------------------------------------
    # Complete experiment
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        if self.running:
            return

        if self.workflow.pipeline.prepared is None:
            self.queue_notice(
                "Prepare the dataset before running the experiment.",
                "warning",
            )
            return

        self.running = True
        self.running_stage = "Complete experiment"
        self._touch()

        asyncio.create_task(
            self._run_all_worker()
        )

    async def _run_all_worker(self) -> None:
        try:
            payload = await asyncio.to_thread(
                self.workflow.run_all
            )

            if isinstance(payload, dict):
                self.result_payload = payload
                self.fuzzy_result = (
                    payload.get("fuzzy")
                    if isinstance(payload.get("fuzzy"), dict)
                    else self.fuzzy_result
                )

            self.queue_notice(
                "Complete experiment finished."
            )
        except Exception as exc:
            self.queue_notice(
                f"Experiment failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    # ------------------------------------------------------------------
    # Single-record prediction
    # ------------------------------------------------------------------

    def predict_record(self, row_index: int) -> None:
        raw_df = getattr(
            self.workflow.pipeline,
            "raw_dataframe",
            None,
        )

        if raw_df is None:
            self.queue_notice(
                "Load a dataset first.",
                "warning",
            )
            return

        if row_index < 0 or row_index >= len(raw_df):
            self.queue_notice(
                "Selected row is outside the dataset.",
                "negative",
            )
            return

        if self.running:
            return

        self.running = True
        self.running_stage = "Single-record analysis"
        self._touch()

        asyncio.create_task(
            self._predict_worker(row_index)
        )

    async def _predict_worker(
        self,
        row_index: int,
    ) -> None:
        try:
            raw_df = self.workflow.pipeline.raw_dataframe
            row = raw_df.iloc[[row_index]].copy()

            self.prediction_result = await asyncio.to_thread(
                self.workflow.predict,
                row,
            )

            self.queue_notice(
                "Single-record analysis completed."
            )
        except Exception as exc:
            self.queue_notice(
                f"Prediction failed: {exc}",
                "negative",
            )
        finally:
            self.running = False
            self.running_stage = ""
            self._touch()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def export_report(self) -> None:
        try:
            artifact_dir = Path(
                self.workflow.config.artifacts_dir
            )
            artifact_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            report_path = artifact_dir / "experiment_report.json"

            path = self.workflow.export_report(
                report_path
            )

            self.queue_notice(
                f"Report exported: {path}"
            )
        except Exception as exc:
            self.queue_notice(
                f"Report export failed: {exc}",
                "negative",
            )

        self._touch()


@ui.page("/")
def index() -> None:
    state = AppState()
    state.attach_progress_callback()

    with ui.left_drawer(value=True).classes(
        "bg-white border-r"
    ):
        @ui.refreshable
        def sidebar_content() -> None:
            render_sidebar(
                state.current_page,
                state.change_page,
                stage_status=state.workflow.stage_status,
                on_run_all=state.run_all,
            )

        state.refresh_sidebar = sidebar_content.refresh
        sidebar_content()

    with ui.column().classes(
        "w-full min-h-screen bg-gray-50"
    ):
        with ui.column().classes("w-full"):
            @ui.refreshable
            def shell_content() -> None:
                with ui.row().classes(
                    "w-full items-center justify-between px-6 py-4 bg-white border-b sticky top-0 z-10"
                ):
                    with ui.column().classes("gap-0"):
                        ui.label(APP_TITLE).classes(
                            "text-xl font-bold"
                        )
                        ui.label(
                            "Research Demonstration Workspace"
                        ).classes(
                            "text-xs text-gray-500"
                        )

                    with ui.row().classes(
                        "items-center gap-3"
                    ):
                        if state.running:
                            ui.spinner(size="sm")
                            ui.label(
                                state.running_stage
                            ).classes(
                                "text-sm text-gray-600"
                            )
                        else:
                            ui.badge(
                                "Ready",
                                color="positive",
                            )

            state.refresh_shell = shell_content.refresh
            shell_content()

        with ui.column().classes(
            "w-full p-6"
        ):
            @ui.refreshable
            def main_content() -> None:
                render_workspace(state)

            state.refresh_main = main_content.refresh
            main_content()

    def client_tick() -> None:
        # All UI operations are performed from the client timer.
        if state.pending_notice is not None:
            notice = state.drain_notice()
            if notice:
                message, kind = notice
                ui.notify(message, type=kind)

        if state.state_version != state.rendered_version:
            state.refresh()

    ui.timer(0.5, client_tick)

    # Light periodic update while a worker is running.
    ui.timer(
        1.0,
        lambda: state.refresh() if state.running else None,
    )


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host=HOST,
        port=PORT,
        title=APP_TITLE,
        reload=False,
        show=False,
    )
