#!/bin/bash
# 量化因子挖掘流水线 · 双击启动器
# 双击本文件 → 浏览器自动打开操作台（http://localhost:8501）
cd "$(dirname "$0")"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
streamlit run app.py
