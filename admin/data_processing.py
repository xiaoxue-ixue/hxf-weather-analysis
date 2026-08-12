import io

import pandas as pd
import streamlit as st


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


# 查看数据基本信息(处理前)
def inspect_data(data: pd.DataFrame) -> None:
    st.markdown("##### 数据的摘要信息")
    # 查看数据的摘要信息
    buffer = io.StringIO()
    data.info(buf=buffer)
    # 把数据的摘要信息展示到页面中（代码形式展示）
    st.code(buffer.getvalue())

    st.markdown("##### 类别数据唯一值统计")
    # 类别数据唯一值统计
    # 筛选出类别数据的字段
    columns = [col for col in data.columns if col not in ['date', 'max_temp', 'min_temp']]
    for column in columns:
        # 统计指定列的所有唯一值
        unique_value = data[column].unique()
        # 统计指定列的所有唯一值的数量
        unique_num = data[column].nunique()
        # 把数据展示到页面中（代码形式展示）
        st.code(f"{column} 的唯一值: {unique_value}，数量：{unique_num}")


# 重复值处理
def handle_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    # 先检查重复数据（两行完全一样的数据是重复数据）
    # 两列数据展示，左边处理前的数据量2，右边处理后的数据量
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 重复值检测结果")
        dup_data = [{
            "数据量(处理前)": data.shape[0],  # 获取数据行数量
            "重复值数量": data.duplicated().sum()  # 统计重复数据的数量
        }]
        st.dataframe(dup_data)
    with col2:
        # 重复值处理
        if data.duplicated().sum() > 0:
            # 删除重复值
            data = data.drop_duplicates()
        st.markdown("##### 重复值处理结果")
        dup_data = [{
            "数据量(处理后)": data.shape[0],  # 获取数据行数量
            "重复值数量": data.duplicated().sum()  # 统计重复数据的数量
        }]
        st.dataframe(dup_data)
    return data


# 日期特征处理
def handle_date(data: pd.DataFrame) -> pd.DataFrame:
    col1, col2 = st.columns([1, 3])
    # 展示处理前的日期特征
    with col1:
        st.markdown("##### 日期特征(处理前)")
        st.dataframe(data['date'], hide_index=True)

    # 把date列转为日期时间类型，errors='coerce'表示解析失败设置为空值
    data['date'] = pd.to_datetime(data['date'], format='%Y年%m月%d日', errors='coerce')
    # 提取日期相关特征（年、月、日、周、年积日、季节），每个特征作为一个新列并入df中
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['week'] = data['date'].dt.isocalendar().week
    data['dayofyear'] = data['date'].dt.dayofyear

    def get_season(month) -> str:
        if pd.isna(month):
            return '未知'
        if month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        elif month in [9, 10, 11]:
            return '秋季'
        elif month in [12, 1, 2]:
            return '冬季'

    data['season'] = data['month'].apply(get_season)
    # 展示处理后的日期特征
    with col2:
        st.markdown("##### 日期特征(处理后)")
        fields = ['date', 'year', 'month', 'day', 'week', 'dayofyear']
        st.dataframe(
            data[fields],
            hide_index=True,
            column_config={
                'date': st.column_config.DateColumn(format='YYYY-MM-DD')
            }
        )

    return data


# 风力等级简化
def simplify_wind_force(data: pd.DataFrame) -> pd.DataFrame:
    col1, col2 = st.columns(2)
    # 使用第一列展示原数据
    with col1:
        st.markdown("##### 风力等级(处理前)")
        st.dataframe(data[['wind_force_day', 'wind_force_night']], hide_index=True)

    # 根据蒲福风等级分类
    def get_wind_category(wind_force) -> str:
        if pd.isna(wind_force):
            return '微风'
        wf = str(wind_force)
        if any(c in wf for c in ['1', '2', '3', '微']):
            return '微风'
        elif any(c in wf for c in ['4', '5']):
            return '清风'
        elif any(c in wf for c in ['6', '7']):
            return '强风'
        elif '8' in wf:
            return '大风'
        return '微风'

    data['wind_force_day'] = data['wind_force_day'].apply(get_wind_category)
    data['wind_force_night'] = data['wind_force_night'].apply(get_wind_category)
    # 使用第二列展示处理后数据
    with col2:
        st.markdown("##### 风力等级(处理后)")
        st.dataframe(data[['wind_force_day', 'wind_force_night']], hide_index=True)

    return data


# 缺失值处理
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    # 检测缺失值
    def missing_detection(data: pd.DataFrame) -> pd.DataFrame:
        # 计算缺失值数量和缺失率
        missing_count = data.isnull().sum()
        missing_ratio = (missing_count / data.shape[0] * 100).round(4)
        missing_data = pd.DataFrame({
            "列名": missing_count.index,
            "缺失数量": missing_count.values,
            "缺失率": missing_ratio
        })
        return missing_data

    col1, col2 = st.columns(2)
    # 使用第一列展示缺失值情况（处理前）
    with col1:
        st.markdown("##### 缺失值情况(处理前)")
        st.dataframe(missing_detection(data), hide_index=True)

    # 缺失值处理（使用向前填充方法处理）
    columns = [col for col in data.columns if col not in ['city', 'date']]
    for column in columns:
        # 向前填充缺失值
        # data[column] = data[column].fillna(method='ffill')
        data[column] = data[column].ffill()
    # 如果city或date为空，则直接删除
    data.dropna(axis=0, subset=['city', 'date'], how='any', inplace=True)

    # 使用第er列展示缺失值情况（处理后）
    with col2:
        st.markdown("##### 缺失值情况(处理后)")
        st.dataframe(missing_detection(data), hide_index=True)

    return data


# 温度格式标准化
def temp_format(data: pd.DataFrame) -> pd.DataFrame:
    col1, col2 = st.columns([2, 3])
    # 使用第一列展示气温数据（处理前）
    with col1:
        st.markdown("##### 气温数据（处理前）")
        st.dataframe(data[['max_temp', 'min_temp']], hide_index=True)

    # 气温格式处理
    data['max_temp'] = data['max_temp'].apply(lambda t: str(t).replace('℃', '').strip() if pd.notna(t) else 0).astype(int)
    data['min_temp'] = data['min_temp'].apply(lambda t: str(t).replace('℃', '').strip() if pd.notna(t) else 0).astype(int)
    data['avg_temp'] = (data['max_temp'] + data['min_temp']) / 2
    # 使用第二列展示气温数据（处理后）
    with col2:
        st.markdown("##### 气温数据（处理后）")
        st.dataframe(data[['max_temp', 'min_temp', 'avg_temp']], hide_index=True)

    return data


# 数据类型转换
def data_type_convert(data: pd.DataFrame) -> pd.DataFrame:
    # 查看数据类型
    def view_type(data: pd.DataFrame) -> pd.DataFrame:
        data_type = data.dtypes.reset_index()
        data_type.columns = ['字段名', '数据类型']
        return data_type

    col1, col2 = st.columns([2, 3])
    # 使用第一列展示数据类型（处理前）
    with col1:
        st.markdown("##### 数据类型（处理前）")
        st.dataframe(view_type(data), hide_index=True)

    # 数据类型转换
    data['week'] = data['week'].astype(int)

    # 使用第一列展示数据类型（处理前）
    with col2:
        st.markdown("##### 数据类型（处理后）")
        st.dataframe(view_type(data), hide_index=True)

    return data

# 把数据保存到数据库
def save_to_sqlite(data: pd.DataFrame, table: str) -> None:
    # 创建数据库连接引擎
    # engine = create_engine(config['sqlite_url'])
    engine = st.connection(name="weather", type='sql').engine
    # 把df数据保存到数据库
    count = data.to_sql(
        name=table,  # 表名
        con=engine,  # 数据库连接引擎
        if_exists='replace',  # 数据存在的保存方式，append表示追加，replace表示替换
        index=False,  # 不写入索引列
        chunksize=1000  # 数据分批写入，避免数据量过大报错
    )
    st.success(f"成功保存 {count} 条数据到 {table} 表")


# Streamlit页面标签展示
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "数据基本信息(处理前)",
    "重复值处理",
    "日期特征处理",
    "风力等级简化",
    "缺失值处理",
    "温度格式标准化",
    "数据类型转换",
    "数据基本信息(处理后)",
    "清洗后数据"
])

df = load_data_from_sqlite(st.secrets.db.original_data_table)

with tab1:
    inspect_data(df)
with tab2:
    df = handle_duplicates(df)
with tab3:
    df = handle_date(df)
with tab4:
    df = simplify_wind_force(df)
with tab5:
    df = handle_missing_values(df)
with tab6:
    df = temp_format(df)
with tab7:
    df = data_type_convert(df)
with tab8:
    inspect_data(df)
with tab9:
    save_to_sqlite(df,st.secrets.db.cleaned_data_table)
    # 展示清洗后的数据
    st.subheader("清洗后数据预览")
    st.dataframe(df, width='stretch')

