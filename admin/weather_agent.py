import os
import re
import pandas as pd
import streamlit as st
import requests
from typing import Optional

# ---------- 配置读取 ----------
def get_db_table_names():
    db_config = st.secrets.get("db", {})
    raw_table = db_config.get("original_data_table", "original_weather_data")
    clean_table = db_config.get("cleaned_data_table", "cleaned_weather_data")
    return raw_table, clean_table

# ---------- 云端调用适配器 ----------
class CloudModelAdapter:
    @staticmethod
    def call_dify(question: str, context: str) -> Optional[str]:
        api_url = os.getenv("DIFY_API_URL") or st.secrets.get("dify", {}).get("endpoint")
        api_key = os.getenv("DIFY_API_KEY") or st.secrets.get("dify", {}).get("api_key")
        if not api_url or not api_key:
            return None
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "inputs": {"question": question, "context": context},
                "response_mode": "blocking",
                "user": "weather_user"
            }
            resp = requests.post(api_url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("answer")
            else:
                st.warning(f"Dify 返回错误码 {resp.status_code}")
                return None
        except Exception as e:
            st.warning(f"Dify 调用异常: {e}")
            return None

    @staticmethod
    def call_qwen(question: str, context: str) -> Optional[str]:
        api_key = st.secrets.get("qwen", {}).get("api_key")
        endpoint = st.secrets.get("qwen", {}).get("endpoint")
        if not api_key or not endpoint:
            return None
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "qwen-plus",
                "input": {
                    "messages": [
                        {"role": "system", "content": f"你是一个天气助手，参考以下数据上下文回答：\n{context}"},
                        {"role": "user", "content": question}
                    ]
                },
                "parameters": {"result_format": "message"}
            }
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("output", {}).get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content")
                return data.get("result") or data.get("answer")
            else:
                st.warning(f"Qwen 返回错误码 {resp.status_code}")
                return None
        except Exception as e:
            st.warning(f"Qwen 调用异常: {e}")
            return None

    @classmethod
    def get_cloud_answer(cls, question: str, context: str) -> Optional[str]:
        ans = cls.call_dify(question, context)
        if ans is not None:
            return ans
        ans = cls.call_qwen(question, context)
        if ans is not None:
            return ans
        return None

# ---------- 核心 Agent ----------
class WeatherChatAgent:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        self.raw_df, self.clean_df = self._load_data()
        self.context = self._build_context()
        self.example_questions = [
            "平均气温是多少？",
            "最高气温和最低气温分别是多少？",
            "最常见的天气是什么？",
            "各城市的平均气温如何？",
            "每月气温变化怎么样？",
            "数据有多少条记录？",
            "今天天气怎么样？"
        ]

    def _load_data(self):
        raw_table, clean_table = get_db_table_names()
        try:
            conn = st.connection("weather", type="sql")
            engine = conn.engine
            raw_df = pd.read_sql_table(raw_table, con=engine) if raw_table else pd.DataFrame()
            clean_df = pd.read_sql_table(clean_table, con=engine) if clean_table else pd.DataFrame()
        except Exception as e:
            st.error(f"读取数据库失败: {e}")
            raw_df = pd.DataFrame()
            clean_df = pd.DataFrame()
        return raw_df, clean_df

    def _build_context(self) -> str:
        lines = []
        lines.append("当前天气数据摘要：")
        lines.append(f"- 原始数据记录数：{len(self.raw_df)}")
        if not self.raw_df.empty and 'date' in self.raw_df.columns:
            min_d = self.raw_df['date'].min()
            max_d = self.raw_df['date'].max()
            lines.append(f"- 原始数据时间范围：{min_d} 至 {max_d}")
        lines.append(f"- 清洗后记录数：{len(self.clean_df)}")
        if not self.clean_df.empty and 'date' in self.clean_df.columns:
            lines.append(f"- 清洗后最新日期：{self.clean_df['date'].max()}")
        if not self.clean_df.empty and 'avg_temp' in self.clean_df.columns:
            avg = self.clean_df['avg_temp'].mean()
            lines.append(f"- 清洗后平均气温：{avg:.1f}℃")
        if not self.clean_df.empty and 'type_day' in self.clean_df.columns:
            mode_series = self.clean_df['type_day'].mode()
            if not mode_series.empty:
                mode = mode_series.iloc[0]
            else:
                mode = "未知"
            lines.append(f"- 最常见白天天气：{mode}")
        if not self.clean_df.empty and 'city' in self.clean_df.columns:
            cities = self.clean_df['city'].unique()
            if len(cities) > 0:
                lines.append(f"- 涉及城市：{', '.join(cities[:5])}{'...' if len(cities)>5 else ''}")
        return "\n".join(lines)

    def _local_answer(self, question: str) -> Optional[str]:
        """本地问答引擎，返回答案或 None"""
        q = question.lower().strip()

        # 平均气温
        if "平均气温" in q or "平均温度" in q:
            if 'avg_temp' not in self.clean_df.columns or self.clean_df['avg_temp'].dropna().empty:
                return None
            avg = self.clean_df['avg_temp'].mean()
            return f"📊 清洗后数据中，平均气温为 **{avg:.1f}℃**。"

        # 最高气温
        if "最高气温" in q or "最高温度" in q:
            if 'avg_temp' not in self.clean_df.columns:
                return None
            max_val = self.clean_df['avg_temp'].max()
            if pd.isna(max_val):
                return None
            return f"🔥 清洗后数据中，最高气温为 **{max_val:.1f}℃**。"

        # 最低气温
        if "最低气温" in q or "最低温度" in q:
            if 'avg_temp' not in self.clean_df.columns:
                return None
            min_val = self.clean_df['avg_temp'].min()
            if pd.isna(min_val):
                return None
            return f"❄️ 清洗后数据中，最低气温为 **{min_val:.1f}℃**。"

        # 最常见天气
        if "最常见天气" in q or "最多天气" in q or "主要天气" in q:
            for col in ['type_day', 'type_night', 'weather']:
                if col in self.clean_df.columns and not self.clean_df[col].dropna().empty:
                    mode_series = self.clean_df[col].mode()
                    if not mode_series.empty:
                        mode_val = mode_series.iloc[0]
                        return f"☁️ 清洗后数据中，最常见的天气类型为 **{mode_val}**（基于 `{col}` 列统计）。"
            return None

        # 按城市查询（若指定城市则返回该城市，否则返回所有城市TOP5）
        if "城市" in q and "平均" in q:
            if 'city' not in self.clean_df.columns or 'avg_temp' not in self.clean_df.columns:
                return None
            cities_in_data = self.clean_df['city'].unique()
            matched_city = None
            for city in cities_in_data:
                if city in q:
                    matched_city = city
                    break
            if matched_city:
                avg = self.clean_df[self.clean_df['city'] == matched_city]['avg_temp'].mean()
                return f"🌆 {matched_city}的平均气温为 **{avg:.1f}℃**。"
            else:
                # 未指定具体城市，返回前5
                city_avg = self.clean_df.groupby('city')['avg_temp'].mean().round(1)
                if city_avg.empty:
                    return None
                top_cities = city_avg.sort_values(ascending=False).head(5)
                lines = ["🌆 **各城市平均气温（前5）**："]
                for city, temp in top_cities.items():
                    lines.append(f"- {city}: {temp}℃")
                return "\n".join(lines)

        # 月份统计
        if "月份" in q or "每月" in q or "月平均" in q:
            if 'date' not in self.clean_df.columns or 'avg_temp' not in self.clean_df.columns:
                return None
            df = self.clean_df.dropna(subset=['date', 'avg_temp'])
            if df.empty:
                return None
            month_avg = df.groupby(df['date'].dt.month)['avg_temp'].mean().round(1)
            lines = ["📅 **各月平均气温**："]
            for m, temp in month_avg.items():
                lines.append(f"- {m}月: {temp}℃")
            return "\n".join(lines)

        # 记录数
        if "多少条" in q or "记录数" in q or "数据量" in q:
            raw_count = len(self.raw_df)
            clean_count = len(self.clean_df)
            return f"📋 原始数据共有 **{raw_count}** 条记录，清洗后为 **{clean_count}** 条。"

        # 查询特定日期或最新日期
        if "今天" in q or "昨天" in q or "某天" in q or "日期" in q:
            if 'date' not in self.clean_df.columns or 'avg_temp' not in self.clean_df.columns:
                return None
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            match = re.search(date_pattern, q)
            if match:
                target_date = match.group()
                row = self.clean_df[self.clean_df['date'] == target_date]
                if not row.empty:
                    avg = row['avg_temp'].iloc[0]
                    weather = row.get('type_day', ['未知']).iloc[0]
                    return f"📅 {target_date} 天气：{weather}，平均气温 {avg:.1f}℃。"
                else:
                    return f"未找到 {target_date} 的数据，请检查日期格式或选择其他日期。"
            else:
                # 没有指定具体日期，返回最新日期的天气
                if not self.clean_df['date'].dropna().empty:
                    latest = self.clean_df.loc[self.clean_df['date'].idxmax()]
                    date_str = latest['date'].strftime('%Y-%m-%d')
                    avg = latest['avg_temp']
                    weather = latest.get('type_day', '未知')
                    return f"📅 最新数据日期 {date_str} 的天气：{weather}，平均气温 {avg:.1f}℃。"
                else:
                    return None

        # 按天气类型统计
        if "晴天" in q or "雨天" in q or "阴天" in q or "多云" in q:
            if 'type_day' not in self.clean_df.columns:
                return None
            weather_types = ['晴天', '雨天', '阴天', '多云', '雪天']
            target_type = None
            for wt in weather_types:
                if wt in q:
                    target_type = wt
                    break
            if target_type:
                count = len(self.clean_df[self.clean_df['type_day'] == target_type])
                total = len(self.clean_df)
                if total > 0:
                    return f"☁️ 在清洗数据中，{target_type} 出现 **{count}** 次，占比 {count / total:.1%}。"
                else:
                    return f"没有 {target_type} 的记录。"
            else:
                return "请指定天气类型（如晴天、雨天等）。"

        return None

    def get_answer(self, user_input: str) -> str:
        """主入口：先本地，再云端，最后兜底"""
        # 1. 尝试本地回答
        local_ans = self._local_answer(user_input)
        if local_ans is not None:
            return local_ans

        # 2. 尝试云端回答
        cloud_ans = CloudModelAdapter.get_cloud_answer(user_input, self.context)
        if cloud_ans is not None and cloud_ans.strip():
            return cloud_ans.strip()

        # 3. 兜底提示
        return (
            "🤔 抱歉，我暂时无法回答这个问题。\n\n"
            "💡 您可以尝试：\n"
            "- 从下方的“快速提问”中选择一个问题\n"
            "- 询问具体城市（如“昆明的平均气温”）\n"
            "- 查询某一天（如“2026-07-11的天气”）\n"
            "- 按天气类型统计（如“晴天有多少天”）\n\n"
            "⚙️ 如需更强大的回答，可在 `.streamlit/secrets.toml` 中配置 Dify 或 Qwen 云端模型。"
        )

    def add_message(self, role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})

    def clear_history(self):
        st.session_state.messages.clear()

# ---------- UI 页面 ----------
def main():
    st.title("🌤️ 云南省历史天气查询助手")

    # 初始化 Agent
    if "agent" not in st.session_state:
        st.session_state.agent = WeatherChatAgent()
    agent = st.session_state.agent

    # 侧边栏
    with st.sidebar:
        st.header("管理")
        if st.button("🗑️ 清空对话历史", use_container_width=True):
            agent.clear_history()
            st.rerun()

        st.divider()
        st.caption("数据表配置：")
        raw, clean = get_db_table_names()
        st.caption(f"原始表: `{raw}`")
        st.caption(f"清洗表: `{clean}`")

        # 云端配置状态提示
        st.divider()
        st.caption("云端模型状态：")
        dify_key = os.getenv("DIFY_API_KEY") or st.secrets.get("dify", {}).get("api_key")
        qwen_key = st.secrets.get("qwen", {}).get("api_key")
        if dify_key or qwen_key:
            st.success("✅ 已配置云端模型（可处理开放问题）")
        else:
            st.warning("⚠️ 未配置云端模型，仅支持本地统计问答")
            with st.expander("📖 如何配置？"):
                st.markdown(
                    """
                    在 `.streamlit/secrets.toml` 中添加：
                    ```toml
                    [dify]
                    api_key = "your-dify-api-key"
                    endpoint = "https://api.dify.ai/v1/chat-messages"

                    [qwen]
                    api_key = "your-qwen-api-key"
                    endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
                    """
                )

    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 用户输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        agent.add_message("user", prompt)

        # 获取AI回复
        response = agent.get_answer(prompt)

        # 添加AI回复
        agent.add_message("assistant", response)

        # 刷新页面以显示新消息
        st.rerun()

if __name__ == "__main__":
    main()
