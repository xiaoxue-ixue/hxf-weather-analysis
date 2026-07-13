import random
import time
from typing import Optional, Dict, List

import pandas as pd
import requests
import os
import streamlit as st
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, inspect

# ==================== 请求头配置 =========
# ===========
headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'referer': 'https://www.tianqihoubao.com/lishi/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0',
}

# ==================== 全局配置 ====================
config = {
    'base_url': 'https://www.tianqihoubao.com/lishi/yunnan.htm',  # 云南省历史天气首页
    'table': st.secrets.db.original_data_table,                         # 数据库表名
    'sleep_range': (0.1, 1),                                       # 请求间隔随机范围（秒）
    'sqlite_url': 'sqlite:///./db/weather.db'                     # 初始相对路径（后续会被绝对路径覆盖）
}

# 将相对路径转为绝对路径，并确保 db 目录存在
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(os.path.dirname(BASE_DIR), 'db')
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, 'weather.db')
config['sqlite_url'] = f'sqlite:///{DB_PATH}'   # 自动适配绝对路径

# ==================== CSS 选择器（根据目标网页结构配置） ====================
css = {
    'city_css': 'body > div.main-container > div.content-wrapper > div.main-content > div.citychk > dl > dt > a',
    'year_css': 'body > div.main-container > div.content-wrapper > div.main-content > div.card-body',
    'month_css': 'ul > li > a',
    'day_css': 'body > div.main-container > div.content-wrapper > div.main-content > div.table-responsive > table > tbody > tr',
    'date_css': 'td:nth-child(1) > a',
    'type_css': 'td:nth-child(2)',
    'max_temp_css': 'td:nth-child(3) > span.temp-high',
    'min_temp_css': 'td:nth-child(3) > span.temp-low',
    'dire_wind_css': 'td:nth-child(4)'
}


# ==================== 工具函数 ====================

# 创建会话
def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(headers)
    return session


# 发送请求获取响应
# [修改] 增加超时参数和重试机制，防止因网络超时返回 None
def safe_request(session: requests.Session, url: str, retries: int = 3, timeout: int = 30) -> Optional[requests.Response]:
    # 每次发送请求前随机延迟，避免触发反爬机制
    time.sleep(random.uniform(*config['sleep_range']))
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)  # 发送get请求，设置超时
            response.raise_for_status()  # 如果返回的状态码不是2xx（200），主动抛出异常
            response.encoding = response.apparent_encoding  # 自动猜测网站的真实编码
            return response
        except requests.exceptions.Timeout:
            print(f"⏱️ 请求超时（尝试 {attempt+1}/{retries}）：{url}")
            time.sleep(2)  # 等待2秒后重试
        except requests.exceptions.RequestException as e:
            print(f"请求发送失败: {str(e)}")
            break  # 非超时错误不再重试
    return None


# ==================== 解析函数 ====================

# 提取全部的城市的天气链接，提取各个城市的天气链接数据
# markup：网页源码
def parse_city_links(response: requests.Response) -> List[Dict[str, str]]:
    # [修改] 增加防御性检查，避免传入 None
    if response is None:
        return []
    # response.text：在响应中获取网页源码
    soup = BeautifulSoup(markup=response.text, features='lxml')
    city_tags = soup.select(css['city_css'])
    return [
        {
            'name': city_tag.select_one('b').get_text(),  # 提取城市名称
            'url': "http://www.tianqihoubao.com" + city_tag.get('href')  # 提取城市链接
        }
        for city_tag in city_tags
    ]


# 循环获取城市链接，发送请求，获取每个城市每年每个月的天气链接，提取每个城市每年每个月的天气链接数据
def parse_month_links(response: requests.Response) -> List[str]:
    month_links = []
    soup = BeautifulSoup(markup=response.text, features='lxml')
    year_tags = soup.select(css['year_css'])
    for year_tag in year_tags:
        month_tags = year_tag.select(css['month_css'])
        if month_tags:
            for month_tag in month_tags:
                href = month_tag.get('href')
                if 'lishi' in href:
                    month_url = f"http://www.tianqihoubao.com{href}"  # 提取月份天气链接地址
                else:
                    month_url = f"http://www.tianqihoubao.com/lishi/{href}"
                month_links.append(month_url)
    return month_links


# 再循环每个月的城市天气链接，得到每个月的天气数据，提取数据
def parse_day_weather(response: requests.Response, city_name: str) -> pd.DataFrame:
    # 保存天气数据的列表
    weather_data = []
    soup = BeautifulSoup(markup=response.text, features='lxml')
    day_tags = soup.select(css['day_css'])
    for day_tag in day_tags:
        # 提取日期
        date_tag = day_tag.select_one(css['date_css'])
        date = date_tag.get_text(strip=True) if date_tag else None
        # 提取天气类型
        type_tag = day_tag.select_one(css['type_css'])
        weather_type = type_tag.get_text(strip=True) if type_tag else None
        # 白天天气类型
        type_day = weather_type.split('/')[0].strip() if weather_type else None
        # 夜间天气类型
        type_night = weather_type.split('/')[1].strip() if weather_type else None
        # 提取气温
        max_temp_tag = day_tag.select_one(css['max_temp_css'])
        max_temp = max_temp_tag.get_text(strip=True) if max_temp_tag else None
        min_temp_tag = day_tag.select_one(css['min_temp_css'])
        min_temp = min_temp_tag.get_text(strip=True) if min_temp_tag else None
        # 提取风向风力
        dire_wind_tag = day_tag.select_one(css['dire_wind_css'])
        dire_wind = dire_wind_tag.get_text(strip=True) if dire_wind_tag else None
        # 白天风向
        direction_day = dire_wind.split('/')[0].strip().split(' ')[0] if dire_wind else None
        # 白天风力等级
        wind_force_day = dire_wind.split('/')[0].strip().split(' ')[1] if dire_wind else None
        # 夜间风向
        direction_night = dire_wind.split('/')[1].strip().split(' ')[0] if dire_wind else None
        # 夜间风力等级
        wind_force_night = dire_wind.split('/')[1].strip().split(' ')[1] if dire_wind else None
        print(
            f"城市：{city_name},"
            f"日期：{date},"
            f"白天天气类型：{type_day},"
            f"夜间天气类型：{type_night},"
            f"最高气温：{max_temp},"
            f"最低气温：{min_temp},"
            f"白天风向：{direction_day},"
            f"白天风力等级：{wind_force_day},"
            f"夜间风向：{direction_night},"
            f"夜间风力等级：{wind_force_night}"
        )
        # 把数据追加保存到列表中
        weather_data.append({
            'city': city_name,
            'date': date,
            'type_day': type_day,
            'type_night': type_night,
            'max_temp': max_temp,
            'min_temp': min_temp,
            'direction_day': direction_day,
            'wind_force_day': wind_force_day,
            'direction_night': direction_night,
            'wind_force_night': wind_force_night,
        })
    return pd.DataFrame(weather_data)


# ==================== 数据库操作（去重 + 保存） ====================

# 优化点3：如果代码重复执行，不重复采集数据
def is_data_exist(table: str, city_name: str, year_month_str: str) -> bool:
    # 判断表是否存在，不存在则返回False，需要爬取数据
    engine = create_engine(config['sqlite_url'])
    # 获取sqlite数据库中的所有表名
    table_list = inspect(engine).get_table_names()
    # 检查表名是否存在
    if table not in table_list:
        print(f"表 {table} 未创建，即将开始采集数据")
        return False

    sql = f"""
    select city, date
    from {table}
    where city = :city
    and substr(replace(replace(date,'年',''),'月',''),1,6)=:year_month_str
    """
    params = {"city": city_name, "year_month_str": year_month_str}
    df = pd.read_sql(sql, engine, params=params)
    # 如果不为空，说明数据已存在，不用爬取，返回True
    if not df.empty:
        print(f"{city_name}市 {year_month_str} 数据已存在，不用爬取")
        return True
    # 如果为空，说明数据不存在，需要爬取，返回False
    return False


# 把数据保存到数据库
def save_to_sqlite(data: pd.DataFrame, table: str) -> None:
    engine = create_engine(config['sqlite_url'])
    # 把df数据保存到数据库
    data.to_sql(
        name=table,  # 表名
        con=engine,  # 数据库连接引擎
        if_exists='append',  # 数据存在的保存方式，append表示追加，replace表示替换
        index=False,  # 不写入索引列
        chunksize=1000  # 数据分批写入，避免数据量过大报错
    )


# ==================== 主采集逻辑 ====================

# 主方法，用于控制执行逻辑
def main():
    # 创建会话
    session = create_session()
    # 获取云南省历史天气的页面响应
    province_response = safe_request(session, config['base_url'])
    # [修改] 检查省级页面是否请求成功，失败则直接返回
    if province_response is None:
        print("❌ 无法获取云南省历史天气首页，采集终止")
        return
    # 解析网页，提取各个城市的城市名称和城市天气链接地址
    city_list = parse_city_links(province_response)
    print(f"共有 {len(city_list)} 城市：{[city['name'] for city in city_list]}")
    # 循环获取city的链接数据，并发送请求，获取每个城市的网页响应
    for city in city_list:
        # 获取城市历史天气的页面响应
        city_response = safe_request(session, city['url'])
        # [修改] 检查城市页面是否请求成功，失败则跳过该城市
        if city_response is None:
            print(f"⚠️ 获取 {city['name']} 页面失败，跳过该城市")
            continue
        # 解析各个城市的天气网页，提取一个城市所有的月份天气链接地址
        month_links = parse_month_links(city_response)
        # 循环获取一个城市每个月的天气链接
        for month_link in month_links:
            # 优化点1：从地址中提取出年份和月份
            year_month_str = month_link.split('/')[-1].replace('.html', '')
            year = int(year_month_str[:4])
            month = int(year_month_str[4:])
            print(f"正在采集 {city['name']}市 {year}年{month}月的天气数据")
            # 优化点2：过滤2015年1月之前的数据，不采集
            if year < 2015:
                continue
            # 优化点3：如果代码重复执行，不重复采集数据
            if is_data_exist(config['table'], city['name'], year_month_str):
                continue
            # 并发送请求，获取每个月的天气链接响应
            month_response = safe_request(session, month_link)
            # [修改] 检查月份页面是否请求成功，失败则跳过该月份
            if month_response is None:
                print(f"⚠️ 获取 {city['name']} {year}年{month}月页面失败，跳过")
                continue
            df = parse_day_weather(month_response, city['name'])
            # 保存数据到数据库
            save_to_sqlite(df, config['table'])


# ==================== Streamlit 界面 ====================

# 与streamlit整合在一起

# 加载数据
def load_data_from_sqlite(table_name: str) -> pd.DataFrame:
    engine = create_engine(config['sqlite_url'])
    try:
        data = pd.read_sql_table(table_name, con=engine)
        return data
    except Exception:
        return pd.DataFrame()   # 表不存在时返回空DataFrame，页面显示“暂无数据”


# 在侧边栏新增按钮
button = st.sidebar.button(label="开始采集数据", type="secondary", use_container_width=True)

# 如果按钮被点击
if button:
    # 增加一个状态提示组件
    with st.spinner("数据正在采集中..."):
        # 执行数据采集逻辑
        main()
    # 数据采集完成后的提示
    st.success("数据采集完成")

# 读取采集后的数据
data = load_data_from_sqlite(config['table'])
# 添加二级标题
st.subheader("原始数据预览")
# 展示采集后的数据
st.dataframe(data, use_container_width=True, height=500)