"""同步 Job 执行器：完整记录状态，并优先读取确定性缓存。"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Mapping, TypeVar
from uuid import uuid4

from qfm.jobs.models import JobRecord, JobStatus, build_cache_key, utc_now
from qfm.jobs.store import JobStore


T = TypeVar("T")


@dataclass(frozen=True)
class JobResult(Generic[T]):
    job: JobRecord
    value: T


class JobService:
    def __init__(self, root: Path | str = "data_cache/jobs"):
        self.store = JobStore(root)

    @staticmethod
    def _read_result(path: Path):
        with path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - 本机私有缓存，不接收外部上传

    @staticmethod
    def _write_result(path: Path, value) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def run(self, job_type: str, request: Mapping, compute: Callable[[], T]) -> JobResult[T]:
        cache_key = build_cache_key(job_type, request)
        job = self.store.save_job(JobRecord.create(job_type, request, cache_key))
        job = self.store.update_job(job, status=JobStatus.RUNNING.value, started_at=utc_now())
        result_path = self.store.result_path(cache_key)

        if result_path.exists():
            try:
                value = self._read_result(result_path)
            except (OSError, EOFError, pickle.PickleError, AttributeError, ImportError, ValueError, TypeError):
                value = None
            else:
                job = self.store.update_job(
                    job,
                    status=JobStatus.SUCCESS.value,
                    finished_at=utc_now(),
                    result_path=str(result_path),
                    cache_hit=True,
                )
                return JobResult(job=job, value=value)

        try:
            value = compute()
            self._write_result(result_path, value)
        except Exception as exc:
            job = self.store.update_job(
                job,
                status=JobStatus.FAILED.value,
                finished_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        job = self.store.update_job(
            job,
            status=JobStatus.SUCCESS.value,
            finished_at=utc_now(),
            result_path=str(result_path),
            cache_hit=False,
        )
        return JobResult(job=job, value=value)

