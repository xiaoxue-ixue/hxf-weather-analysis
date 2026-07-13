import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import seaborn as sns
import category_encoders as ce
import joblib
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

# 把文件所在的路径加入系统环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# 导入ai图表解读自定义模块
import ai_chart_interpretation as aci

# 解决中文显示为乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置图表自动布局
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True


# 从sqlite数据库加载原始数据
@st.cache_data
def load_data_from_sqlite(table_name: str) -> pd.DataFrame:
    # 创建数据库查询连接引擎
    engine = st.connection(name="weather", type='sql').engine
    # 使用pandas中根据表名查询整个表数据的方法
    data = pd.read_sql_table(table_name, con=engine)
    # 删除空值，按行删除，至少要有5个有效字段，否则删除
    data.dropna(axis=0, thresh=5, inplace=True)
    # 重置索引
    data.reset_index(drop=True, inplace=True)
    return data


# 特征工程
def feature_engineering(data: pd.DataFrame):
    # 选择特征和目标字段
    features = ['city', 'year', 'month', 'day', 'week', 'season', 'dayofyear']
    target = ['min_temp', 'max_temp']
    st.markdown("##### 1.特征与目标数据")
    col1, col2 = st.columns([7, 2])
    with col1:
        st.markdown("##### 特征数据")
        st.dataframe(data[features], hide_index=True, width='stretch')
    with col2:
        st.markdown("##### 目标数据")
        st.dataframe(data[target], hide_index=True, width='stretch')
    # 特征数据处理：特征编码，季节（标签编码）、城市（目标编码）
    # 数据准备（特征+目标）
    data = data[features + target + ['avg_temp']]
    #######################################
    # 获取编码前的city
    cities = data['city']
    #######################################
    # 季节（标签编码）
    data['season'] = data['season'].map({'春季': 0, '夏季': 1, '秋季': 2, '冬季': 3})
    # 城市（目标编码）：pip install category_encoders
    data = ce.TargetEncoder(cols=['city']).fit_transform(data, data['avg_temp'])
    #######################################
    # 获取编码后的city
    encoded_cities = data['city']
    # 保存目标编码后的数据（city及对应的编码值）
    encoding_value_df = pd.DataFrame({'city': cities, 'encoding_value': encoded_cities}).drop_duplicates()
    encoding_value_df.to_csv('city_encoding_value.csv', index=False)
    #######################################
    # 展示编码后的特征数据
    st.markdown("##### 2.特征数据（编码后）")
    st.dataframe(data[features], hide_index=True, width='stretch')
    # 特征数据
    X = data[features]
    # 目标数据
    y_min = data['min_temp']
    y_max = data['max_temp']
    # 数据集划分：训练集 / 测试集，划分比例（7:3 或 8:2）
    X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test = train_test_split(
        X, y_min, y_max, test_size=0.2, random_state=42, shuffle=True)

    # 展示数据集划分结果
    st.markdown("##### 3.数据集划分结果")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.metric(label="训练集", value=f"{X_train.shape[0]} / {X_train.shape[0] / X.shape[0] * 100:.2f}%")
    with col2:
        with st.container(border=True):
            st.metric(label="测试集", value=f"{X_test.shape[0]} / {X_test.shape[0] / X.shape[0] * 100:.2f}%")

    # 返回划分结果
    return X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test


# 保存图片，返回图片路径、图片连接
def save_image(fig: plt.Figure, file_name: str) -> (str, str):
    # 获取项目根路径
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # 修复：使用 os.path.join
    image_path = os.path.join(project_root, "static", f"{file_name}.png")
    # 确保static目录存在
    static_dir = os.path.dirname(image_path)
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    # 保存图片，bbox_inches：裁剪掉图片周围的空白边框
    fig.savefig(image_path, bbox_inches='tight')
    # 构建图片访问url
    image_url = f"http://127.0.0.1:{st.get_option('server.port')}/app/static/{file_name}.png?t={int(time.time())}"
    return image_path, image_url


# 误差分布直方图
def error_distribution_histogram(data: pd.Series, title: str) -> None:
    # 创建画布和绘图区域
    fig, ax = plt.subplots(figsize=(8, 5))
    # 绘制误差分布直方图
    sns.histplot(data=data, bins=30, ax=ax)
    # 绘制0误差线
    ax.axvline(x=0, color='r', linestyle='--', label='0误差')
    # 设置标题
    ax.set_title(title)
    # 设置x轴和y轴标签
    ax.set_xlabel('误差', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    # 显示图例
    ax.legend()
    # 设置边框为浅灰色
    ax.spines[['top', 'right', 'bottom', 'left']].set_color('#cccccc')
    # 自动调整页面布局
    plt.tight_layout()
    # 展示图表
    image_path, image_url = save_image(fig, title)
    st.image(image_path, width='stretch')
    # ai解读图表
    if st.session_state.is_ai:
        aci.chart_interpretation(image_path, image_url)
    # 释放资源
    plt.close(fig)


# 真实值与预测值趋势变化折线图
def true_pred_plot(y_true: List, y_pred: List, title: str) -> None:
    # 创建画布和绘图区域
    fig, ax = plt.subplots(figsize=(10, 5))
    # 绘制折线图
    ax.plot(y_true, label='真实值', marker='o', color='#2E86AB')
    ax.plot(y_pred, label='预测值', marker='s', color='#F24236')
    # 设置标题
    ax.set_title(title)
    # 设置x轴和y轴标签
    ax.set_xlabel('序号', fontsize=12)
    ax.set_ylabel('温度', fontsize=12)
    # 显示图例
    ax.legend()
    # 设置边框为浅灰色
    ax.spines[['top', 'right', 'bottom', 'left']].set_color('#cccccc')
    # 自动调整页面布局
    plt.tight_layout()
    # 展示图表
    image_path, image_url = save_image(fig, title)
    st.image(image_path, width='stretch')
    # ai解读图表
    if st.session_state.is_ai:
        aci.chart_interpretation(image_path, image_url)
    # 释放资源
    plt.close(fig)


# 特征重要性条形图
def barh_chart(data: pd.DataFrame, col: str, title: str) -> None:
    # 把数据根据值降序排序
    data = data.sort_values(by=col, ascending=False)
    # 创建画布和绘图区域
    fig, ax = plt.subplots(figsize=(8, 5))
    # 绘制条形图
    sns.barplot(data=data, x=col, y='feature', ax=ax, palette='Set3')
    # 设置标题
    ax.set_title(title)
    # 设置x轴和y轴标签
    ax.set_xlabel('重要性', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    # 设置边框为浅灰色
    ax.spines[['top', 'right', 'bottom', 'left']].set_color('#cccccc')
    # 自动调整页面布局
    plt.tight_layout()
    # 展示图表
    image_path, image_url = save_image(fig, title)
    st.image(image_path, width='stretch')
    # ai解读图表
    if st.session_state.is_ai:
        aci.chart_interpretation(image_path, image_url)
    # 释放资源
    plt.close(fig)


# 模型构建（模型选择、训练、预测）与评估
def model_build_evaluation(X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test):
    st.markdown("### 随机森林气温预测模型构建与评估")
    button = st.button(label="点击构建与评估模型", type="secondary", width='stretch')
    if button:
        # 最低气温预测模型选择和训练
        with st.spinner("正在训练最低气温预测模型..."):
            # 优化后的模型参数
            model_min = RandomForestRegressor(
                n_estimators=50,    # 减少树的数量
                max_depth=10,       # 限制树的最大深度
                min_samples_split=5, # 增加最小分裂样本数
                min_samples_leaf=2,  # 增加最小叶节点样本数
                random_state=42
            )
            # 模型训练
            model_min.fit(X_train, y_min_train)
        st.success("最低气温预测模型训练完成！")

        # 最高气温预测模型选择和训练
        with st.spinner("正在训练最高气温预测模型..."):
            # 优化后的模型参数
            model_max = RandomForestRegressor(
                n_estimators=50,    # 减少树的数量
                max_depth=10,       # 限制树的最大深度
                min_samples_split=5, # 增加最小分裂样本数
                min_samples_leaf=2,  # 增加最小叶节点样本数
                random_state=42
            )
            # 模型训练
            model_max.fit(X_train, y_max_train)
        st.success("最高气温预测模型训练完成！")

        # 模型预测
        y_min_pred = model_min.predict(X_test)
        y_max_pred = model_max.predict(X_test)
        st.success("模型预测完成！")

        # 保存模型
        with st.spinner("正在保存模型..."):
            # 获取项目根路径
            project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            # 确保model目录存在
            model_dir = os.path.join(project_root, "model")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # 统一模型文件命名
            min_model_path = os.path.join(model_dir, "model_min_temp.pkl")
            max_model_path = os.path.join(model_dir, "model_max_temp.pkl")

            # 使用压缩选项保存模型
            joblib.dump(model_min, min_model_path, compress=3)
            joblib.dump(model_max, max_model_path, compress=3)

        st.success(f"最低气温和最高气温预测模型已保存到 {model_dir} 目录下")

        # 预测数据和真实数据的对比表格
        st.markdown("##### 1.预测数据和真实数据对比")
        result_df = pd.DataFrame({
            "最低温(真实)": y_min_test.values,
            "最低温(预测)": y_min_pred,
            "最低温(误差)": (y_min_test.values - y_min_pred).round(2),
            "最高温(真实)": y_max_test.values,
            "最高温(预测)": y_max_pred,
            "最高温(误差)": (y_max_test.values - y_max_pred).round(2)
        })
        st.dataframe(result_df, hide_index=True, width='stretch')

        # 模型可视化评估
        # 误差分布直方图
        st.markdown("##### 2.误差分布直方图")
        col1, col2 = st.columns(2)
        with col1:
            error_distribution_histogram(result_df["最低温(误差)"], "最低温误差分布直方图")
        with col2:
            error_distribution_histogram(result_df["最高温(误差)"], "最高温误差分布直方图")

        # 真实值与预测值趋势变化折线图
        st.markdown("##### 3.真实值与预测值对比折线图")
        col1, col2 = st.columns(2)
        with col1:
            true_pred_plot(
                result_df["最低温(真实)"].values[:80],
                result_df["最低温(预测)"].values[:80],
                "最低温真实值与预测值对比折线图"
            )
        with col2:
            true_pred_plot(
                result_df["最高温(真实)"].values[:80],
                result_df["最高温(预测)"].values[:80],
                "最高温真实值与预测值对比折线图"
            )

        # 特征重要性分析
        features = X_train.columns.tolist()
        # 最低气温预测模型特征重要性数据
        feature_importances_min = model_min.feature_importances_
        # 最高气温预测模型特征重要性数据
        feature_importances_max = model_max.feature_importances_
        feature_importances_df = pd.DataFrame({
            "feature": features,
            "importances_min": feature_importances_min,
            "importances_max": feature_importances_max
        })
        st.markdown("##### 4.特征重要性分析")
        st.markdown("##### 4.1 特征重要性数据")
        st.dataframe(feature_importances_df, hide_index=True, width='stretch')
        st.markdown("##### 4.2 特征重要性数据可视化")
        col1, col2 = st.columns(2)
        with col1:
            barh_chart(feature_importances_df, 'importances_min', '最低温特征重要性条形图')
        with col2:
            barh_chart(feature_importances_df, 'importances_max', '最高温特征重要性条形图')

        # 模型评估：计算回归模型核心评价指标
        # 最低气温回归模型核心评价指标：R^2, MAE, MSE, RMSE
        mae_min = mean_absolute_error(y_min_test, y_min_pred)
        mse_min = mean_squared_error(y_min_test, y_min_pred)
        rmse_min = np.sqrt(mse_min)
        r2_min = r2_score(y_min_test, y_min_pred)
        # 最高气温回归模型核心评价指标：R^2, MAE, MSE, RMSE
        mae_max = mean_absolute_error(y_max_test, y_max_pred)
        mse_max = mean_squared_error(y_max_test, y_max_pred)
        rmse_max = np.sqrt(mse_max)
        r2_max = r2_score(y_max_test, y_max_pred)
        st.markdown("##### 5.模型核心评价指标")
        metric_df = pd.DataFrame({
            '模型': ['最低气温预测模型', '最高气温预测模型'],
            'MAE': [mae_min, mae_max],
            'MSE': [mse_min, mse_max],
            'RMSE': [rmse_min, rmse_max],
            'R^2': [r2_min, r2_max]
        })
        st.dataframe(metric_df, hide_index=True, width='stretch')


# 加载气温预测模型
@st.cache_resource
def load_model():
    """加载预训练的气温预测模型"""
    try:
        # 定义模型路径
        project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        model_dir = os.path.join(project_root, "model")

        # 检查model目录是否存在
        if not os.path.exists(model_dir):
            st.error("模型目录不存在！请先在【模型构建与评估】页面完成模型训练。")
            return None, None

        # 定义模型文件路径
        min_model_path = os.path.join(model_dir, "model_min_temp.pkl")
        max_model_path = os.path.join(model_dir, "model_max_temp.pkl")

        # 检查文件是否存在
        if not os.path.exists(max_model_path):
            st.error(f"最高气温模型文件不存在: {max_model_path}")
            return None, None
        if not os.path.exists(min_model_path):
            st.error(f"最低气温模型文件不存在: {min_model_path}")
            return None, None

        # 加载模型
        model_max = joblib.load(max_model_path)
        model_min = joblib.load(min_model_path)
        return model_min, model_max
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None, None


# 模型应用：用户选择城市和指定日期，预测指定日期的气温范围
def model_application():
    st.markdown("""
        云南省天气气温预测
    """, unsafe_allow_html=True)

    # 获取项目根路径，使用绝对路径读取 CSV
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(project_root, "city_encoding_value.csv")

    # 增加文件存在性检查，防止文件未生成导致崩溃
    if not os.path.exists(csv_path):
        st.warning("⚠️ 未找到城市编码文件！请先在【模型构建与评估】页面完成模型训练。")
        return

    city_encoding = pd.read_csv(csv_path)
    city_encoding_dict = dict(zip(city_encoding['city'], city_encoding['encoding_value']))

    # 季节编码字典
    def get_season(month) -> int:
        if month in [3, 4, 5]:
            return 0
        elif month in [6, 7, 8]:
            return 1
        elif month in [9, 10, 11]:
            return 2
        elif month in [12, 1, 2]:
            return 3

    # 模型预测的特征构建（城市编码、日期处理、季节判断）
    def build_model_input(city: str, date: datetime.date) -> pd.DataFrame:
        month = date.month
        # 构建预测特征df
        feature_df = pd.DataFrame({
            'city': [city_encoding_dict.get(city)],
            'year': [date.year],
            'month': [month],
            'day': [date.day],
            'week': [date.weekday()],
            'season': [get_season(month)],
            'dayofyear': [date.timetuple().tm_yday]
        })
        return feature_df

    city_col, date_col = st.columns(2)
    # 城市下拉选择框
    with city_col:
        city = st.selectbox(label="请选择城市", options=city_encoding['city'])
    with date_col:
        date = st.date_input(label="请选择日期")
    feature = build_model_input(city, date)

    # 检查模型文件是否存在
    model_dir = os.path.join(project_root, "model")
    min_model_path = os.path.join(model_dir, "model_min_temp.pkl")
    max_model_path = os.path.join(model_dir, "model_max_temp.pkl")

    if not os.path.exists(min_model_path) or not os.path.exists(max_model_path):
        st.warning("⚠️ 未检测到气温预测模型！请先在【模型构建与评估】页面完成模型训练。")
        return

    button = st.button(label="开始预测气温", type="primary", width='stretch')
    # 加载气温预测模型
    model_min, model_max = load_model()

    if button:
        if model_min is None or model_max is None:
            st.error("模型加载失败，请先在【模型构建与评估】页面完成模型训练。")
            return

        with st.spinner("正在预测气温..."):
            try:
                pred_min = model_min.predict(feature)[0]
                pred_max = model_max.predict(feature)[0]
                st.markdown('---')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('预测最低气温', f"{pred_min:.0f}℃")
                with col2:
                    st.metric('预测最高气温', f"{pred_max:.0f}℃")
                st.success(f"{city}市{date}的气温预测：{pred_min:.0f}℃ ~ {pred_max:.0f}℃")
            except Exception as e:
                st.error(f"预测过程中发生错误: {str(e)}")


# 加载数据
df = load_data_from_sqlite(st.secrets.db.cleaned_data_table)

# 页面布局
tab1, tab2, tab3 = st.tabs(["特征工程", "模型构建与评估", "模型应用"])
with tab1:
    X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test = feature_engineering(df)
with tab2:
    model_build_evaluation(X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test)
with tab3:
    model_application()
