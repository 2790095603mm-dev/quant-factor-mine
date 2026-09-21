"""基于 JSON 与 CSV 的本地研究账本。"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from qfm.data.catalog import REQUIRED_BINDING_FIELDS
from qfm.research.models import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    LoadedResearchRun,
    ResearchProject,
    ResearchRun,
)


def _factor_version_index(definitions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """从因子定义快照中抽取"版本索引"，写进 manifest 供复现时核对。

    完整定义（含源码）单独存 factors.json，manifest 只留轻量索引。
    """
    if not definitions:
        return []
    index: list[dict[str, Any]] = []
    for item in definitions:
        if not isinstance(item, Mapping) or not item.get("name"):
            continue
        index.append({
            "name": str(item["name"]),
            "version": item.get("version"),
            "source_hash": item.get("source_hash"),
        })
    return index


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
        *,
        status: str = STATUS_COMPLETED,
        tags: tuple[str, ...] | list[str] = (),
        code_version: dict[str, Any] | None = None,
        factor_definitions: list[dict[str, Any]] | None = None,
        require_binding: bool = False,
    ) -> ResearchRun:
        self.get_project(project_id)
        if require_binding:
            missing = [key for key in REQUIRED_BINDING_FIELDS if not data_snapshot.get(key)]
            if missing:
                raise ValueError(
                    "研究运行缺少数据版本绑定: " + ", ".join(missing)
                )
        artifacts = {
            "nav": "nav.csv",
            "benchmark_nav": "benchmark_nav.csv",
            "weights": "weights.csv",
            "yearly_performance": "yearly_performance.csv",
            "trades": "trades.csv",
        }
        if constraint_history is not None and not constraint_history.empty:
            artifacts["constraint_history"] = "constraint_history.csv"
        factor_versions = tuple(_factor_version_index(factor_definitions))
        if factor_definitions:
            artifacts["factor_definitions"] = "factors.json"
        run = ResearchRun.create(
            project_id, name, config, data_snapshot, summary, artifacts,
            status=status, tags=tags, code_version=code_version,
            factor_versions=factor_versions,
        )
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
            if "factor_definitions" in artifacts and factor_definitions:
                self._write_json(
                    temporary / artifacts["factor_definitions"],
                    {"factors": [dict(item) for item in factor_definitions]},
                )
            self._write_json(temporary / "manifest.json", run.to_dict())
            temporary.replace(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return run

    def save_failed_run(
        self,
        project_id: str,
        name: str,
        config: dict[str, Any],
        error: str,
        *,
        tags: tuple[str, ...] | list[str] = (),
        data_snapshot: dict[str, Any] | None = None,
    ) -> ResearchRun:
        """记录一次失败的运行。

        失败运行同样落盘：否则"这条因子在这个参数下跑不出来"这类信息会永久丢失，
        而它恰恰是复现时最容易踩的坑。
        """
        self.get_project(project_id)
        message = str(error).strip() or "未知错误"
        run = ResearchRun.create(
            project_id,
            name,
            config,
            data_snapshot or {},
            {"失败原因": message},
            {"error": "error.txt"},
            status=STATUS_FAILED,
            tags=tags,
            error=message,
        )
        target = self.runs_dir / run.id
        temporary = self.runs_dir / f".{run.id}.{uuid4().hex}.tmp"
        if target.exists():
            raise FileExistsError(f"研究运行已存在: {run.id}")
        try:
            temporary.mkdir()
            (temporary / "error.txt").write_text(message + "\n", encoding="utf-8")
            self._write_json(temporary / "manifest.json", run.to_dict())
            temporary.replace(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return run

    def delete_run(self, run_id: str) -> None:
        """删除一次运行的全部产物（不可撤销）。"""
        directory = self.runs_dir / run_id
        if not directory.is_dir():
            raise KeyError(f"研究运行不存在: {run_id}")
        shutil.rmtree(directory)

    def load_run_config(self, run_id: str) -> dict[str, Any]:
        """取回一次运行的参数，供"载入到表单"重放实验。

        返回 config 的深拷贝，并附带 run_id / name / factor_versions，
        使界面可以提示"当前表单来自某次历史运行"。
        """
        directory = self.runs_dir / run_id
        if not directory.is_dir():
            raise KeyError(f"研究运行不存在: {run_id}")
        run = self._read_run(directory)
        restored = json.loads(json.dumps(run.config, ensure_ascii=False))
        restored["_run_id"] = run.id
        restored["_run_name"] = run.name
        restored["_factor_versions"] = [dict(item) for item in run.factor_versions]
        restored["_data_snapshot"] = {
            "data_start": run.data_snapshot.get("data_start"),
            "data_end": run.data_snapshot.get("data_end"),
            "pool": run.data_snapshot.get("pool"),
            "universe_fingerprint": run.data_snapshot.get("universe_fingerprint"),
            "market_fingerprint": (run.data_snapshot.get("fingerprints") or {}).get("market"),
            "dataset_id": run.data_snapshot.get("dataset_id"),
            "dataset_version": run.data_snapshot.get("dataset_version"),
            "universe_id": run.data_snapshot.get("universe_id"),
            "universe_version": run.data_snapshot.get("universe_version"),
            "legacy_unbound": run.legacy_unbound,
        }
        return restored

    def load_factor_definitions(self, run_id: str) -> list[dict[str, Any]]:
        """读取运行归档的因子定义快照（v1 运行没有该产物，返回空列表）。"""
        directory = self.runs_dir / run_id
        if not directory.is_dir():
            raise KeyError(f"研究运行不存在: {run_id}")
        run = self._read_run(directory)
        if "factor_definitions" not in run.artifacts:
            return []
        path = directory / run.artifacts["factor_definitions"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        factors = payload.get("factors")
        return [dict(item) for item in factors] if isinstance(factors, list) else []

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

    def list_all_runs(self) -> list[tuple[ResearchProject, ResearchRun]]:
        """列出所有项目的可读取运行，供跨项目 Strategy Compare 使用。"""
        pairs: list[tuple[ResearchProject, ResearchRun]] = []
        for project in self.list_projects():
            runs, _warnings = self.list_runs(project.id)
            pairs.extend((project, run) for run in runs)
        return sorted(pairs, key=lambda item: item[1].created_at, reverse=True)

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
