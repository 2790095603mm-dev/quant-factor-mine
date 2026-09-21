"""Job manifest 与缓存产物的本地存储。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from qfm.jobs.models import JobRecord


class JobStore:
    def __init__(self, root: Path | str = "data_cache/jobs"):
        self.root = Path(root)

    @property
    def manifests_dir(self) -> Path:
        path = self.root / "manifests"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_dir(self) -> Path:
        path = self.root / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def result_path(self, cache_key: str) -> Path:
        return self.results_dir / f"{cache_key}.pkl"

    def save_job(self, job: JobRecord) -> JobRecord:
        target = self.manifests_dir / f"{job.job_id}.json"
        temporary = target.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(job.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return job

    def update_job(self, job: JobRecord, **changes: Any) -> JobRecord:
        return self.save_job(replace(job, **changes))

    def get_job(self, job_id: str) -> JobRecord:
        path = self.manifests_dir / f"{job_id}.json"
        if not path.exists():
            raise KeyError(f"Job 不存在: {job_id}")
        try:
            return JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            raise ValueError(f"Job manifest 损坏: {job_id}") from exc

    def list_jobs(self, limit: int | None = None) -> list[JobRecord]:
        jobs: list[JobRecord] = []
        for path in self.manifests_dir.glob("job_*.json"):
            try:
                jobs.append(JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, ValueError, KeyError):
                continue
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs[:limit] if limit is not None else jobs

