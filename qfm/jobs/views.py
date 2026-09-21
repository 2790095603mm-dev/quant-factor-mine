"""Job Center 页面。"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from qfm.jobs.models import JOB_TYPES, JobStatus
from qfm.jobs.store import JobStore


def _duration(started: str, finished: str) -> float | None:
    if not started or not finished:
        return None
    try:
        return (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        return None


def render_job_center(store: JobStore) -> None:
    st.markdown(
        '<div class="qfm-sig"><h1>任务中心</h1>'
        '<div class="sub">所有因子计算、分析、多因子和回测都留下状态与缓存命中记录。</div></div>',
        unsafe_allow_html=True,
    )
    jobs = store.list_jobs()
    counts = {status.value: sum(job.status == status.value for job in jobs) for status in JobStatus}
    for column, status in zip(st.columns(4), JobStatus):
        column.metric(status.value, counts[status.value])
    c1, c2 = st.columns(2)
    job_type = c1.selectbox("任务类型", ["全部", *JOB_TYPES], key="job_filter_type")
    status = c2.selectbox("状态", ["全部", *[item.value for item in JobStatus]], key="job_filter_status")
    filtered = [
        job for job in jobs
        if (job_type == "全部" or job.job_type == job_type)
        and (status == "全部" or job.status == status)
    ]
    if not filtered:
        st.info("还没有符合条件的任务。")
        return
    frame = pd.DataFrame([
        {
            "Job ID": job.job_id,
            "类型": job.job_type,
            "状态": job.status,
            "缓存": "命中" if job.cache_hit else "计算",
            "耗时（秒）": _duration(job.started_at, job.finished_at),
            "创建时间": job.created_at[:19].replace("T", " "),
            "错误": job.error,
        }
        for job in filtered[:200]
    ])
    st.dataframe(frame.style.format({"耗时（秒）": "{:.2f}"}, na_rep="—"), hide_index=True, use_container_width=True)
    chosen = st.selectbox("查看任务请求", [job.job_id for job in filtered], key="job_detail")
    selected = next(job for job in filtered if job.job_id == chosen)
    st.json(selected.request)
    if selected.error:
        st.error(selected.error)
