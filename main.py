import streamlit as st
import sys
import os
from pathlib import Path

import toml

# 添加项目根目录到系统路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 设置页面配置
st.set_page_config(
    page_title="云南省历史天气数据分析",
    page_icon="🌤️",
    layout="wide"
)

# 定义页面
pages = {
    "📋 管理模块": [
        st.Page("./admin/data_collection.py", title="数据采集", icon="📥"),
        st.Page("./admin/data_processing.py", title="数据处理", icon="🔄")
    ],
    "📈 分析模块": [
        st.Page("./admin/data_analysis.py", title="数据分析", icon="📊"),
        st.Page("./admin/data_temperature_forecast.py", title="气温预测", icon="🌡️")
    ],
    "💬 智能助手": [
        st.Page("./admin/weather_agent.py", title="云南省历史天气查询助手", icon="🤖")
    ]
}

# 加载配置文件
try:
    with open('.streamlit/secrets.toml') as f:
        secrets = toml.load(f)
except FileNotFoundError:
    st.error("未找到配置文件，请确保 .streamlit/secrets.toml 文件存在")
    st.stop()


# 创建导航
nav = st.navigation(
    pages,
    position="sidebar",
    expanded=False
)

# 运行导航
nav.run()
