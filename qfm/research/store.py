"""基于 JSON 与 CSV 的本地研究账本。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from qfm.research.models import LoadedResearchRun, ResearchProject, ResearchRun


class ResearchStore:
    """管理单机研究项目和已完成回测的可审计产物。"""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def projects_dir(self) -> Path:
        path = self.root / "projects"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def runs_dir(self) -> Path:
        path = self.root / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    @staticmethod
    def _write_csv(frame: pd.DataFrame | pd.Series, path: Path, *, index_label: str | None = None) -> None:
        frame.to_csv(path, index=index_label is not None, index_label=index_label, encoding="utf-8")

    def _write_project(self, project: ResearchProject) -> None:
        target = self.projects_dir / f"{project.id}.json"
        temporary = target.with_suffix(".tmp")
        try:
            self._write_json(temporary, project.to_dict())
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def create_project(self, name: str, description: str = "") -> ResearchProject:
        project = ResearchProject.create(name, description)
        self._write_project(project)
        return project

    def list_projects(self) -> list[ResearchProject]:
        projects: list[ResearchProject] = []
        for path in self.projects_dir.glob("project_*.json"):
            try:
                projects.append(ResearchProject.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError, ValueError):
                continue
        return sorted(projects, key=lambda project: project.updated_at, reverse=True)

    def get_project(self, project_id: str) -> ResearchProject:
        path = self.projects_dir / f"{project_id}.json"
        if not path.exists():
            raise KeyError(f"研究项目不存在: {project_id}")
        try:
            return ResearchProject.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            raise ValueError(f"研究项目记录损坏: {project_id}") from exc

    def save_run(
        self,
        project_id: str,
        name: str,
        config: dict[str, Any],
        data_snapshot: dict[str, Any],
        summary: dict[str, Any],
        nav: pd.Series,
        benchmark_nav: pd.Series,
        weights: pd.DataFrame,
        yearly_performance: pd.DataFrame,
        trades: pd.DataFrame,
        constraint_history: pd.DataFrame | None = None,
    ) -> ResearchRun:
        self.get_project(project_id)
        artifacts = {
            "nav": "nav.csv",
            "benchmark_nav": "benchmark_nav.csv",
            "weights": "weights.csv",
            "yearly_performance": "yearly_performance.csv",
            "trades": "trades.csv",
        }
        if constraint_history is not None and not constraint_history.empty:
            artifacts["constraint_history"] = "constraint_history.csv"
        run = ResearchRun.create(project_id, name, config, data_snapshot, summary, artifacts)
        target = self.runs_dir / run.id
        temporary = self.runs_dir / f".{run.id}.{uuid4().hex}.tmp"
        if target.exists():
            raise FileExistsError(f"研究运行已存在: {run.id}")

        try:
            temporary.mkdir()
            self._write_csv(nav.rename("nav"), temporary / artifacts["nav"], index_label="date")
            self._write_csv(
                benchmark_nav.rename("benchmark_nav"),
                temporary / artifacts["benchmark_nav"],
                index_label="date",
            )
            self._write_csv(weights, temporary / artifacts["weights"], index_label="factor")
            self._write_csv(yearly_performance, temporary / artifacts["yearly_performance"])
            self._write_csv(trades, temporary / artifacts["trades"])
            if "constraint_history" in artifacts and constraint_history is not None:
                self._write_csv(constraint_history, temporary / artifacts["constraint_history"])
            self._write_json(temporary / "manifest.json", run.to_dict())
            temporary.replace(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return run

    def _read_run(self, directory: Path) -> ResearchRun:
        manifest = directory / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError("缺少 manifest.json")
        return ResearchRun.from_dict(json.loads(manifest.read_text(encoding="utf-8")))

    def list_runs(self, project_id: str) -> tuple[list[ResearchRun], list[str]]:
        self.get_project(project_id)
        runs: list[ResearchRun] = []
        warnings: list[str] = []
        for directory in self.runs_dir.glob("run_*"):
            if not directory.is_dir():
                continue
            try:
                run = self._read_run(directory)
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                warnings.append(f"无法读取运行 {directory.name}: {exc}")
                continue
            if run.project_id == project_id:
                runs.append(run)
        return sorted(runs, key=lambda run: run.created_at, reverse=True), warnings

    @staticmethod
    def _read_series(path: Path, column: str) -> pd.Series:
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
        if column not in frame.columns:
            raise ValueError(f"产物缺少列: {column}")
        series = frame[column]
        series.name = column
        return series

    @staticmethod
    def _read_trades(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, dtype={"stock": str})
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        if "signal_date" in frame.columns:
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        return frame

    @staticmethod
    def _read_constraint_history(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        for column in ("signal_date", "execution_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
        return frame

    def load_run(self, run_id: str) -> LoadedResearchRun:
        directory = self.runs_dir / run_id
        if not directory.is_dir():
            raise KeyError(f"研究运行不存在: {run_id}")
        run = self._read_run(directory)
        artifact = lambda key: directory / run.artifacts[key]
        try:
            weights = pd.read_csv(artifact("weights"), index_col="factor")
            yearly = pd.read_csv(artifact("yearly_performance"))
            constraint_history = (
                self._read_constraint_history(artifact("constraint_history"))
                if "constraint_history" in run.artifacts
                else None
            )
            return LoadedResearchRun(
                run=run,
                nav=self._read_series(artifact("nav"), "nav"),
                benchmark_nav=self._read_series(artifact("benchmark_nav"), "benchmark_nav"),
                weights=weights,
                yearly_performance=yearly,
                trades=self._read_trades(artifact("trades")),
                constraint_history=constraint_history,
            )
        except (KeyError, OSError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"研究运行产物损坏: {run_id}") from exc
