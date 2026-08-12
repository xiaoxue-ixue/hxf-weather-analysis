# 🌤️ 基于Streamlit+AI Agent的云南省天气数据可视化分析系统

> 网络爬虫 → 数据清洗 → 可视化分析 → 机器学习预测 → AI 智能问答，一站式天气数据全流程分析平台

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📖 项目简介

云南省气候类型多样，天气数据对农业种植、旅游出行、灾害预警等领域具有重要参考价值。本项目通过网络爬虫自动采集云南省各城市 2015 年以来的历史天气数据，经过数据清洗和特征工程处理后，提供多维度的可视化分析仪表盘，并基于随机森林算法实现气温预测。同时集成 AI 大模型，支持自然语言问答和图表自动解读，降低数据分析门槛。

### 🎯 核心能力

| 能力 | 技术方案 | 亮点 |
|:---|:---|:---|
| 数据采集 | Requests + BeautifulSoup | 增量爬取，自动去重 |
| 数据处理 | Pandas | 8 步清洗流水线 |
| 可视化分析 | Matplotlib + Seaborn + Plotly | 7 种图表 + 交互仪表盘 |
| 气温预测 | Scikit-learn 随机森林 | 4 项指标评估 + 特征重要性 |
| AI 问答 | Dify + Qwen3 | 自然语言查询天气数据 |
| AI 图表解读 | Qwen-VL 多模态 | 自动生成专业分析报告 |

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    展示层 (Streamlit)                     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │ 首页   │ │ 数据   │ │ 数据   │ │ 数据   │ │ 气温   │ │
│  │ 概览   │ │ 采集   │ │ 处理   │ │ 分析   │ │ 预测   │ │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ │
└──────┼──────────┼──────────┼──────────┼──────────┼──────┘
       │          │          │          │          │
┌──────▼──────────▼──────────▼──────────▼──────────▼──────┐
│                    业务逻辑层                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ 爬虫模块     │ │ 清洗模块     │ │ 可视化模块   │        │
│  │ collection  │ │ processing  │ │ analysis    │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ 预测模块     │ │ AI 问答      │ │ AI 图表解读  │        │
│  │ forecast    │ │ agent       │ │ interpret   │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                      数据层                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ SQLite   │  │ Model    │  │ CSV      │              │
│  │ weather  │  │ *.pkl    │  │ encoder  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
hxf-weather-analysis/
├── main.py                              # 应用入口
├── admin/
│   ├── __init__.py                      # 包初始化
│   ├── home.py                          # 首页概览模块
│   ├── data_collection.py               # 数据采集模块（爬虫）
│   ├── data_processing.py               # 数据处理模块（清洗）
│   ├── data_analysis.py                 # 数据可视化分析模块
│   ├── data_temperature_forecast.py     # 气温预测模块（ML）
│   ├── weather_agent.py                 # AI 智能问答模块
│   └── ai_chart_interpretation.py       # AI 图表解读模块
├── model/                               # 训练好的模型文件
├── static/                              # 图表图片输出
├── db/                                  # SQLite 数据库
├── .streamlit/                          # Streamlit 配置
├── requirements.txt                     # 项目依赖
└── readme.md                            # 项目文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- pip 包管理器

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/xiaoxue-ixue/hxf-weather-analysis.git
cd hxf-weather-analysis

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置 AI 密钥（可选，用于 AI 问答和图表解读）
# 在 .streamlit/ 目录下创建 secrets.toml，填入 Dify API Key

# 6. 启动应用
streamlit run main.py
```

启动后浏览器自动打开 `http://localhost:8501`

---

## 📊 功能模块详解

### 1. 首页概览

项目介绍、功能模块导航、技术栈展示、快速开始指南。

### 2. 数据采集模块

| 功能 | 说明 |
|:---|:---|
| 城市链接解析 | 解析云南省下辖城市列表 |
| 月份链接解析 | 解析每个城市的历史月份数据链接 |
| 天气数据解析 | 解析每日天气信息（日期、天气、气温、风向风力） |
| 增量采集 | 自动跳过已存在的数据，避免重复爬取 |
| 数据存储 | 自动保存到 SQLite 数据库 |

### 3. 数据处理模块

8 步数据清洗流水线：

| 步骤 | 处理内容 | 解决的问题 |
|:---:|:---|:---|
| 1 | 数据基本信息检查 | 了解数据结构和质量 |
| 2 | 重复值处理 | 消除爬虫重复数据 |
| 3 | 日期特征提取 | 提取年/月/日/周/季节 |
| 4 | 天气类型简化 | 归类为 6 大天气类型 |
| 5 | 风力等级标准化 | 统一为弱风/中风/强风/大风 |
| 6 | 缺失值填充 | 向前填充法处理气温缺失 |
| 7 | 温度格式标准化 | 去除单位符号，转整数 |
| 8 | 数据类型转换 | 统一字段类型，优化内存 |

### 4. 数据可视化分析模块

| 分析类型 | 图表类型 | 展示内容 |
|:---|:---|:---|
| 描述性统计 | 表格 | 均值、标准差、分位数 |
| 气温分布 | 直方图 + 核密度图 | 最高/最低气温分布 |
| 气温趋势 | 折线图 | 按日期展示气温变化 |
| 风力分布 | 柱状图 | 日间/夜间风力等级 |
| 风向分布 | 条形图 | 日间/夜间风向 |
| 天气类型 | 词云图 | 天气类型频率 |
| 综合仪表盘 | 多图表组合 | 关键指标 + 全维度分析 |

### 5. 气温预测模块

基于随机森林回归算法：

| 阶段 | 功能 | 说明 |
|:---|:---|:---|
| 特征工程 | 7 个特征选择 | 城市、年、月、日、周、季节、年积日 |
| 数据集划分 | 80/20 分割 | 训练集 80%，测试集 20% |
| 模型训练 | 随机森林回归 | 分别训练最高温和最低温模型 |
| 模型评估 | 4 项指标 | MAE、MSE、RMSE、R² |
| 模型应用 | 实时预测 | 输入城市和日期返回预测结果 |
| 误差分析 | 可视化 | 误差直方图 + 真实值/预测值对比 |
| 特征重要性 | 排序图 | 展示各特征对预测的贡献度 |

### 6. AI 智能问答模块

- 基于 Dify + Qwen3 云端大模型
- 自动获取数据库天气数据作为上下文
- 支持自然语言查询，如「昆明最热的月份是几月？」
- 保存对话历史记录
- 提供示例问题快速访问

### 7. AI 图表解读模块

- 集成 Qwen-VL 多模态视觉模型
- 自动识别图表类型
- 生成专业解读报告（基础信息 + 核心分析 + 结论）

---

## 🔧 配置说明

### 数据库配置

项目使用 SQLite，数据库文件自动创建在 `db/weather.db`。

### AI 配置（可选）

在 `.streamlit/secrets.toml` 中配置：

```toml
[connections.weather]
dialect = "sqlite"
url = "sqlite:///db/weather.db"

[config]
original_weather_data = "original_weather_data"
cleaned_weather_data = "cleaned_weather_data"
weather_featured = "weather_featured"
```

---

## 📝 使用说明

| 步骤 | 操作 | 说明 |
|:---:|:---|:---|
| 1 | 数据采集 | 点击「数据采集」→「开始采集」，等待爬虫完成 |
| 2 | 数据处理 | 点击「数据处理」，查看 8 步清洗结果，保存清洗数据 |
| 3 | 数据分析 | 点击「数据分析」，选择城市/年份/月份，查看图表 |
| 4 | 气温预测 | 点击「气温预测」→ 训练模型 → 输入城市和日期预测 |
| 5 | AI 问答 | 点击「智能问答」，用自然语言提问 |
| 6 | 数据导出 | 在仪表盘页面点击「导出 CSV」下载数据 |

---

## 📦 技术栈

| 分类 | 技术 | 用途 |
|:---|:---|:---|
| Web 框架 | Streamlit | 交互式 Web 应用 |
| 数据处理 | Pandas | 数据清洗与分析 |
| 可视化 | Matplotlib / Seaborn / Plotly | 图表绘制 |
| 词云 | WordCloud | 天气类型词云 |
| 机器学习 | Scikit-learn | 随机森林回归 |
| 特征编码 | Category Encoders | 目标编码 |
| 网络爬虫 | Requests + BeautifulSoup | 天气数据采集 |
| AI 集成 | Dify + Qwen3 / Qwen-VL | 智能问答与图表解读 |
| 数据库 | SQLite | 数据持久化 |

---

## 📜 更新日志

| 版本 | 日期 | 更新内容 |
|:---|:---|:---|
| v1.0 | 2026-06 | 初始版本：数据采集、处理、分析、预测 |
| v1.1 | 2026-06 | 新增 AI 智能问答模块（Dify + Qwen3） |
| v1.2 | 2026-06 | 新增 AI 图表解读功能（Qwen-VL） |
| v1.3 | 2026-08 | 修复 Bug，新增首页概览、CSV 导出功能，优化 README |

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

## 👤 Author

**xiaoxue-ixue**

- GitHub: [@xiaoxue-ixue](https://github.com/xiaoxue-ixue)

---

⭐ 如果这个项目对你有帮助，欢迎 Star！
