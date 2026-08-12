import streamlit as st
import pandas as pd

st.set_page_config(page_title="云南省天气数据分析", page_icon="🌤️", layout="wide")

st.markdown("""
    <h1 style="text-align: center; color: #1E88E5;">🌤️ 云南省历史天气数据可视化分析平台</h1>
    <p style="text-align: center; color: #666; font-size: 18px;">基于 Streamlit + AI Agent 的天气数据采集、处理、分析与预测系统</p>
    <hr style="border: 1px solid #E0E0E0;">
""", unsafe_allow_html=True)

st.markdown("### 📌 功能模块")

col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.markdown("#### 📥 数据采集")
        st.markdown("从天气后报网爬取云南省各城市历史天气数据，支持增量采集和去重")
with col2:
    with st.container(border=True):
        st.markdown("#### 🔄 数据处理")
        st.markdown("数据清洗、特征提取、风力简化、缺失值处理、温度标准化")
with col3:
    with st.container(border=True):
        st.markdown("#### 📊 数据分析")
        st.markdown("气温趋势、风力分布、风向分布、天气类型词云、综合仪表盘")

col4, col5, col6 = st.columns(3)
with col4:
    with st.container(border=True):
        st.markdown("#### 🌡️ 气温预测")
        st.markdown("基于随机森林回归的气温预测模型，支持模型训练评估与实时预测")
with col5:
    with st.container(border=True):
        st.markdown("#### 🤖 AI 智能助手")
        st.markdown("基于大模型的天气数据问答助手，支持本地统计与云端 AI 问答")
with col6:
    with st.container(border=True):
        st.markdown("#### 📖 AI 图表解读")
        st.markdown("通义千问多模态模型自动解读可视化图表，生成专业分析报告")

st.markdown("---")
st.markdown("### 🛠️ 技术栈")
tech_data = pd.DataFrame({
    "分类": ["Web 框架", "数据处理", "可视化", "机器学习", "网络爬虫", "AI 集成", "数据库"],
    "技术": ["Streamlit", "Pandas / NumPy", "Matplotlib / Seaborn / WordCloud", "Scikit-learn (随机森林)", "Requests / BeautifulSoup", "Dify / 通义千问 Qwen-VL", "SQLite / SQLAlchemy"]
})
st.dataframe(tech_data, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 🚀 快速开始")
st.code("streamlit run main.py", language="bash")
st.markdown("启动后在浏览器访问 `http://localhost:8501`")

st.markdown("---")
st.markdown("### 📂 项目结构")
st.code("""
hxf-weather-analysis/
├── main.py                      # 应用入口
├── admin/
│   ├── data_collection.py       # 数据采集模块
│   ├── data_processing.py       # 数据处理模块
│   ├── data_analysis.py         # 数据可视化分析模块
│   ├── data_temperature_forecast.py  # 气温预测模块
│   ├── weather_agent.py         # AI 智能问答模块
│   └── ai_chart_interpretation.py   # AI 图表解读模块
├── model/                       # 训练好的模型文件
├── static/                      # 图表图片
├── .streamlit/                  # Streamlit 配置
├── requirements.txt             # 项目依赖
└── readme.md                    # 项目文档
""", language="bash")