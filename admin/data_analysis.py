import sys
from typing import Literal

import os
import time

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from wordcloud import WordCloud
#把文件所在的路径加入系统环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
#导入ai图表解读自定义模块
import admin.ai_chart_interpretation as aci


# 解决中文显示为乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置图表自动布局
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True


# 从sqlite数据库加载原始数据
@st.cache_data
def load_data_from_sqlite(teble_name: str) -> pd.DataFrame:
    # 创建数据库查询连接引擎
    # engine = create_engine(config['sqlite_url'])
    engine = st.connection(name="weather", type='sql').engine
    # 使用pandas中根据表名查询整个表数据的方法
    data = pd.read_sql_table(teble_name, con=engine)
    # 删除空值，按行删除，至少要有5个有效字段，否则删除
    data.dropna(axis=0, thresh=5, inplace=True)
    # 重置索引
    data.reset_index(drop=True, inplace=True)
    return data


# 保存图片，返回图片路径，图片连接
def save_image(fig: plt.Figure, file_name: str) -> (str, str):
    # 获取项目根路径
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # 构建图片保存路径
    image_path = f"{project_root}/static/{file_name}.png"
    # 保存图片，bbox_inches='tight'：去除图片周围空白
    fig.savefig(image_path, bbox_inches='tight')
    # 构建图片访问url
    image_url = f"http://127.0.0.1:{st.get_option('server.port')}/app/{file_name}.png?t={int(time.time())}"
    return image_path, image_url


# 数据筛选，图和数据转换的开关
def data_filter(data: pd.DataFrame) -> (pd.DataFrame, str):
    # 获取城市的唯一值
    cities = data['city'].unique().tolist()
    # 获取年份的唯一值
    years = data['year'].unique().tolist()
    # 获取月份的唯一值
    months = data['month'].unique().tolist()

    city_col, year_col, month_col, toggle_col = st.columns(4)
    with city_col:
        # 下拉选择框，设置key，后续可通过状态变量访问选择的城市，st.session_state.key的值
        city = st.selectbox(label='城市', options=cities, key='city_name')
        # 过滤数据
        data = data[data['city'] == city]
    with year_col:
        # 年份排序，并设置默认选项
        years = pd.Series(years).sort_values(ascending=False).tolist()
        years.insert(0, '全部')
        # 下拉选择框
        year = st.selectbox(label='年份', options=years, key='year_str')
        # 过滤数据
        if year != '全部':
            data = data[data['year'] == year]
    with month_col:
        # 月份排序，并设置默认选项
        months = pd.Series(months).sort_values().tolist()
        months.insert(0, '全部')
        # 下拉选择框
        month = st.selectbox(label='月份', options=months, key='month_str')
        # 过滤数据
        if month != '全部':
            data = data[data['month'] == month]
    with toggle_col:
        # 定义单选按钮，选择展示数据还是图表
        toggle = st.radio(label='', options=['图表', '数据'], horizontal=True)
    # 返回筛选后的数据和开关的状态
    return data, toggle


# 标题联动方法
def title_linking(title: str) -> str:
    # 标题联动实现
    city = st.session_state.city_name
    year = st.session_state.year_str
    month = st.session_state.month_str
    title_new = f"{city}市{year}年{month}月-{title}"
    return title_new


# 通用数据展示方法，展示类别分布的数据
def show_data(data: pd.Series, lable: str, title: str) -> None:
    title_new = title_linking(title)
    st.markdown(f"##### {title_new}")
    # 进行数据统计
    value_counts = data.value_counts()
    value_counts = value_counts.reset_index()
    value_counts.columns = [lable, '天数']
    # 展示数据
    st.dataframe(value_counts, width='stretch', hide_index=True)


# 气温变化趋势折线图（使用matplotlib和seaborn库绘图）
def temp_line_chart(data: pd.DataFrame, title: str) -> None:
    # 创建画布和绘图区域
    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制折线图
    sns.lineplot(data=data, x='date', y='max_temp', ax=ax, label='最高气温')
    sns.lineplot(data=data, x='date', y='min_temp', ax=ax, label='最低气温')
    # 设置标题
    ax.set_title(title)
    # 设置x轴和y轴标签
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('温度（℃）', fontsize=12)
    # 设置边框为浅灰色
    ax.spines[['top', 'right', 'bottom', 'left']].set_color('#cccccc')
    # 自动调整页面布局
    plt.tight_layout()
    # 展示图表
    # st.pyplot(fig)
    image_path, image_url = save_image(fig, title_linking(title))
    st.image(image_path, width='stretch')
    # ai解读图表
    if st.session_state.is_ai:
        aci.chart_interpretation(image_path, image_url)

    # 释放资源
    plt.close(fig)


# 柱状图（条形图）通用方法（使用matplotlib和seaborn库绘图）
def bar_chart(data: pd.Series, lable: str, title: str, type: Literal['bar', 'barh'] = 'bar') -> None:
    # 准备绘图的数据
    # 进行数据统计，统计后默认会自动降序排序
    value_counts = data.value_counts()
    # 提取类别和频数
    categories = value_counts.index.tolist()
    # 创建画布和绘图区域
    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制柱状图（条形图）
    if type == 'bar':
        sns.countplot(x=data, ax=ax, palette='Set2')  # 绘制柱状图
        # 给柱子添加出现次数
        # ax.patches：获取柱子对象列表，p表示循环的每个柱子对象
        # get_x()：获取x轴左边的坐标
        for p in ax.patches:
            height = int(p.get_height())  # 获取柱子的高度，天数
            # 给柱子添加文本注释
            ax.annotate(
                text=f'{height}',
                xy=(p.get_x() + p.get_width() / 2, height),
                ha='center',
                va='bottom',
                fontweight='bold'
            )
    elif type == 'barh':
        sns.countplot(y=data, ax=ax, palette='Set2', order=categories)  # 绘制条形图
        for p in ax.patches:
            width = int(p.get_width())  # 获取柱子的高度，天数
            # 给柱子添加文本注释
            ax.annotate(
                text=f'{width}',
                xy=(width, p.get_y() + p.get_height() / 2),
                ha='left',
                va='center',
                fontweight='bold'
            )
    # 设置标题
    ax.set_title(title_linking(title))
    # 设置x轴和y轴标签
    if type == 'bar':
        ax.set_xlabel(lable, fontsize=12)
        ax.set_ylabel('天数', fontsize=12)
    elif type == 'barh':
        ax.set_xlabel('天数', fontsize=12)
        ax.set_ylabel(lable, fontsize=12)
    # 设置边框为浅灰色
    ax.spines[['top', 'right', 'bottom', 'left']].set_color('#cccccc')
    # 自动调整页面布局
    plt.tight_layout()
   # 展示图表
    st.pyplot(fig)

    # ai解读图表
    if st.session_state.is_ai:
        # 将内存中的图表保存为本地图片文件
        img_path = f"temp_{title}.png"
        fig.savefig(img_path, bbox_inches='tight')

        # 传入图片路径进行解读 (根据 ai_chart_interpretation 的定义，只需传 image_path)
        aci.chart_interpretation(image_path=img_path)

    # 释放资源
    plt.close(fig)



# 天气类型可视化（词云图）
def word_cloud_chart(data: pd.Series, title: str) -> None:
    # 统计每个类别出现的次数
    word_counts = data.value_counts().to_dict()
    # 创建词云
    wc = WordCloud(
        font_path='simhei.ttf',
        scale=5,  # 字体清晰度，越大字体越清晰
        background_color='white',  # 设置背景颜色
        collocations=False  # 关闭词语搭配
    )
    # 根据word_counts生成词云
    wc.generate_from_frequencies(word_counts)
    # 创建画布和绘图区域
    fig, ax = plt.subplots(figsize=(10, 6))
    # 生成词云图，interpolation：让词云显示更平滑
    ax.imshow(wc, interpolation='bilinear')
    # 关闭坐标轴
    ax.axis('off')
    # 设置标题
    ax.set_title(title_linking(title))
    # 展示图表
    # st.pyplot(fig)
    image_path, image_url = save_image(fig, title_linking(title))
    st.image(image_path, width='stretch')

    # ai解读图表
    if st.session_state.is_ai:
        aci.ai_chart_interpretation(image_path, image_url)
    # 释放资源
    plt.close(fig)


# 显示指标
def show_metric(data: pd.DataFrame) -> None:
    # 数据总量、最低温度、最高温度、平均温度、最常见的天气类型
    total_count = data.shape[0]
    min_temp = data['min_temp'].min()
    max_temp = data['max_temp'].max()
    avg_temp = data['avg_temp'].mean().round(2)
    all_weather_type = pd.concat([data['type_day'], data['type_night']], ignore_index=True)
    most_weather_type = all_weather_type.value_counts().idxmax()

    # 创建5列分别展示5个指标
    cols = st.columns(5)
    with cols[0]:
        with st.container(border=True):
            st.metric(label='数据总量', value=total_count)
    with cols[1]:
        with st.container(border=True):
            st.metric(label='最低温度', value=f"{min_temp}℃")
    with cols[2]:
        with st.container(border=True):
            st.metric(label='最高温度', value=f"{max_temp}℃")
    with cols[3]:
        with st.container(border=True):
            st.metric(label='平均温度', value=f"{avg_temp}℃")
    with cols[4]:
        with st.container(border=True):
            st.metric(label='最常见的天气类型', value=str(most_weather_type))


########################################################

# 单选按钮
page = st.sidebar.radio(
    label='数据可视化分析',
    options=[
        '气温变化趋势分析',
        '风力分布分析',
        '风向分布分析',
        '天气类型分布分析',
        '综合仪表盘'
    ]
)

#在侧边栏设置ai图表解读的开关
aci.ai_toggle()
# 加载清洗后的数据
df = load_data_from_sqlite(st.secrets.db.cleaned_data_table)

if page == '气温变化趋势分析':
    st.markdown("##### 气温变化趋势分析")

    # 添加数据过滤，返回过滤后的数据和开关状态
    df, toggle = data_filter(df)
    # 标题联动实现
    title_new = title_linking('气温变化趋势')
    if toggle == '图表':
        # 展示图表
        temp_line_chart(df, title_new)
    elif toggle == '数据':
        st.markdown(f"##### {title_new}")
        # 展示数据
        df = df[['date', 'max_temp', 'min_temp']]
        df.columns = ['日期', '最高气温', '最低气温']
        st.dataframe(
            df, width='stretch', hide_index=True,
            column_config={
                '日期': st.column_config.DateColumn(format='YYYY-MM-DD')
            }
        )

elif page == '风力分布分析':
    st.markdown("##### 风力分布分析")

    # 添加数据过滤，返回过滤后的数据和开关状态
    df, toggle = data_filter(df)
    if toggle == '图表':
        # 展示图表（柱状图）
        col1, col2 = st.columns(2)
        with col1:
            bar_chart(df['wind_force_day'], '白天风力', '白天风力分布', type='bar')
        with col2:
            bar_chart(df['wind_force_night'], '夜间风力', '夜间风力分布', type='bar')
    elif toggle == '数据':
        col1, col2 = st.columns(2)
        # 展示数据
        with col1:
            show_data(df['wind_force_day'], '白天风力', '白天风力分布')
        with col2:
            show_data(df['wind_force_night'], '夜间风力', '夜间风力分布')


elif page == '风向分布分析':
    st.markdown("##### 风向分布分析")

    # 添加数据过滤，返回过滤后的数据和开关状态
    df, toggle = data_filter(df)
    if toggle == '图表':
        # 展示图表（条形图）
        col1, col2 = st.columns(2)  # 修正拼写：columns
        with col1:
            bar_chart(df['direction_day'], '白天风向', '白天风向分布', type='barh')
        with col2:
            bar_chart(df['direction_night'], '夜间风向', '夜间风向分布', type='barh')
    elif toggle == '数据':
        # 展示数据
        col1, col2 = st.columns(2)  # 修正拼写：columns
        with col1:
            show_data(df['direction_day'], '白天风向', '白天风向分布')
        with col2:
            show_data(df['direction_night'], '夜间风向', '夜间风向分布')

elif page == '天气类型分布分析':
    st.markdown("##### 天气类型分布分析")

    # 添加数据过滤，返回过滤后的数据和开关状态
    df, toggle = data_filter(df)
    if toggle == '图表':
        # 展示图表（词云图）
        col1, col2 = st.columns(2)  # 修正拼写：columns
        with col1:
            word_cloud_chart(df['type_day'], '白天天气类型分布')
        with col2:
            word_cloud_chart(df['type_night'], '夜间天气类型分布')
    elif toggle == '数据':
        # 展示数据
        col1, col2 = st.columns(2)  # 修正拼写：columns
        with col1:
            show_data(df['type_day'], '白天天气类型', '白天天气类型分布')
        with col2:
            show_data(df['type_night'], '夜间天气类型', '夜间天气类型分布')


elif page == '综合仪表盘':
    st.markdown("""
        <h3 style="text-align: center; color:#2c3e50; margin:0; padding:0">
        云南省天气数据可视化分析仪表盘
        </h3>
    """, unsafe_allow_html=True)
    # 添加数据过滤，返回过滤后的数据和开关状态
    df, toggle = data_filter(df)
    show_metric(df)
    if toggle == '图表':
        # 展示图表
        # 第一行的3列：显示白天的数据分布
        col1, col2, col3 = st.columns(3)
        with col1:
            bar_chart(df['wind_force_day'], '白天风力', '白天风力分布', type='bar')
        with col2:
            bar_chart(df['direction_day'], '白天风向', '白天风向分布', type='barh')
        with col3:
            word_cloud_chart(df['type_day'], '白天天气类型分布')
        # 第二行的3列：显示夜间的数据分布
        col4, col5, col6 = st.columns(3)
        with col4:
            bar_chart(df['wind_force_night'], '夜间风力', '夜间风力分布', type='bar')
        with col5:
            bar_chart(df['direction_night'], '夜间风向', '夜间风向分布', type='barh')
        with col6:
            word_cloud_chart(df['type_night'], '夜间天气类型分布')
    elif toggle == '数据':
        # 展示数据
        col1, col2, col3 = st.columns(3)
        with col1:
            show_data(df['wind_force_day'], '白天风力', '白天风力分布')
        with col2:
            show_data(df['direction_day'], '白天风向', '白天风向分布')
        with col3:
            show_data(df['type_day'], '白天天气类型', '白天天气类型分布')
        col4, col5, col6 = st.columns(3)
        with col4:
            show_data(df['wind_force_night'], '夜间风力', '夜间风力分布')
        with col5:
            show_data(df['direction_night'], '夜间风向', '夜间风向分布')
        with col6:
            show_data(df['type_night'], '夜间天气类型', '夜间天气类型分布')
