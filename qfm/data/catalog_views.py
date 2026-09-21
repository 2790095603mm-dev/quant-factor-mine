"""Dataset / Universe 管理页面。"""

from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
import streamlit as st

from qfm.data.catalog import BUILTIN_UNIVERSES, DataCatalog


def _parse_symbols(text: str, uploaded: bytes | None = None) -> list[str]:
    chunks = [text]
    if uploaded:
        decoded = uploaded.decode("utf-8-sig")
        try:
            frame = pd.read_csv(io.StringIO(decoded), dtype=str)
        except pd.errors.ParserError:
            chunks.append(decoded)
        else:
            if len(frame.columns):
                chunks.extend(frame.iloc[:, 0].dropna().astype(str).tolist())
    raw = "\n".join(chunks)
    for separator in (",", "，", ";", "；", "\t"):
        raw = raw.replace(separator, "\n")
    return [item.strip() for item in raw.splitlines() if item.strip()]


def render_data_catalog(catalog: DataCatalog) -> None:
    st.markdown(
        '<div class="qfm-sig"><h1>数据与股票池</h1>'
        '<div class="sub">Dataset 与 Universe 都有明确版本；新实验会绑定版本，而不是只记一个模糊名称。</div></div>',
        unsafe_allow_html=True,
    )
    datasets = catalog.list_datasets()
    universes = catalog.list_universes()
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset 版本", len(datasets))
    c2.metric("已登记股票池版本", len(universes))
    c3.metric("自定义股票池", len({item.universe_id for item in universes if item.universe_id.startswith("custom_")}))

    st.markdown("**标准股票池**")
    st.dataframe(
        pd.DataFrame([
            {"Universe ID": key, "名称": value["name"], "来源": value["source"]}
            for key, value in BUILTIN_UNIVERSES.items()
            if key in {"cn_hs300", "cn_zz500", "cn_zz1000", "cn_all_a"}
        ]),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("创建自定义股票池"):
        name = st.text_input("股票池名称", key="catalog_custom_name", placeholder="例如：新能源龙头观察池")
        symbols_text = st.text_area(
            "股票代码",
            key="catalog_custom_symbols",
            placeholder="每行一个，或用逗号分隔，例如：\n000001\n600000",
        )
        upload = st.file_uploader("也可上传 CSV / TXT（读取第一列）", type=["csv", "txt"])
        if st.button("创建并生成版本", key="catalog_create_universe", use_container_width=True):
            try:
                symbols = _parse_symbols(symbols_text, upload.getvalue() if upload else None)
                created = catalog.create_custom_universe(name, symbols)
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(f"创建失败：{exc}")
            else:
                st.success(
                    f"已创建 {created.name}：{len(created.symbols)} 只股票 · {created.universe_version}"
                )
                st.rerun()

    st.markdown("**已登记 Universe 版本**")
    if universes:
        st.dataframe(
            pd.DataFrame([
                {
                    "Universe ID": item.universe_id,
                    "Universe Version": item.universe_version,
                    "名称": item.name,
                    "来源": item.source,
                    "股票数": len(item.symbols),
                    "更新时间": item.updated_at[:19].replace("T", " "),
                }
                for item in universes
            ]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("尚无已登记版本。第一次运行分析或回测后会自动登记。")

    st.markdown("**Dataset 版本**")
    if datasets:
        st.dataframe(
            pd.DataFrame([
                {
                    "Dataset ID": item.dataset_id,
                    "Dataset Version": item.dataset_version,
                    "来源": item.source,
                    "日期范围": f"{item.start_date[:10]} → {item.end_date[:10]}",
                    "股票数": len(item.symbols),
                    "字段数": len(item.fields),
                    "最近登记": item.last_update[:19].replace("T", " "),
                }
                for item in datasets
            ]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("尚无 Dataset 版本。完成一次因子分析或回测后会自动登记。")
