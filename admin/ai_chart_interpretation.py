import dashscope
import streamlit as st
import os
import base64



# 在 data_temperature_forecast.py 文件的开头添加
if 'is_ai' not in st.session_state:
    st.session_state.is_ai = False

# 控制AI解释图表的开发，初始值False
def ai_toggle() -> None:
    """初始化并显示AI图表解读的开关控件"""
    if 'is_ai' not in st.session_state:
        st.session_state.is_ai = False

    st.sidebar.markdown("---")
    st.session_state.is_ai = st.sidebar.toggle(
        "AI图表解读",
        value=st.session_state.is_ai,
        help='开启图标下方会显示图表解读结果'
    )


def validate_api_config():
    """验证API配置是否正确"""
    try:
        if not hasattr(st.secrets, 'qwen'):
            return False, "未找到API配置"

        config = st.secrets.qwen
        if not config.get('api_key'):
            return False, "API Key未配置"

        if not config.get('workspace_id'):
            return False, "Workspace ID未配置"

        return True, "配置验证通过"
    except Exception as e:
        return False, f"配置验证失败: {str(e)}"


# 系统提示词
SYSTEM_PROMPT = """
【角色设定】
你是一个专业的数据分析师，擅长根据用户上传的图表（条形图，柱状图，折线图，直方图，词云图）进行可视化分析，并给出相应的解读结果。
【任务指令】
1.对云南省历史天气数据可视化图表进行专业的解读。
2.条形图解读：主要关注不同类别之间的数据对比。
3.折线图解读：主要关注数据随时间变化的趋势。
4.直方图解读：主要关注数据分布情况
5.词云图解读：主要关注关键词的大小和重要性。
【限制要求】
1.基于用户的图表来进行分析，紧扣云南气候背景。
2.不要编造结论，只能基于图表数据进行解读，文风简洁专业。
3.严格按照输出格式进行回答，不要添加任何额外内容。字数控制在500字以内。
【输出格式】
####1.图表基础信息
####2.图表核心分析
####3.图表解读结论
"""


def validate_image_path(path):
    """验证图片路径是否有效"""
    if not path:
        return False, "图片路径为空"
    if not os.path.exists(path):
        return False, f"图片文件不存在: {path}"
    if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return False, "不支持的图片格式"
    return True, "路径有效"


def image_to_base64(image_path: str) -> str:
    """将本地图片转为base64 Data URI格式，供多模态API读取"""
    # 获取图片格式
    ext = os.path.splitext(image_path)[1].lower().replace('.', '')
    if ext == 'jpg':
        ext = 'jpeg'

    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/{ext};base64,{base64_str}"


def chart_interpretation(image_path: str, user_prompt: str = '请解读这个图表') -> None:
    """
    使用通义千问模型对图表进行解读

    Args:
        image_path: 图表文件路径
        user_prompt: 用户自定义提示词，默认为'请解读这个图表'
    """
    try:
        # 验证图片路径
        is_valid, message = validate_image_path(image_path)
        if not is_valid:
            st.error(message)
            return

        # 验证API配置
        is_valid, message = validate_api_config()
        if not is_valid:
            st.error(message)
            return

        # 配置API
        workspace_id = st.secrets.qwen.workspace_id
        dashscope.base_http_api_url = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1"

        # 本地图片转base64
        image_base64 = image_to_base64(image_path)

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"text": user_prompt},
                    {"image": image_base64}
                ]
            }
        ]

        # 调用API
        response = dashscope.MultiModalConversation.call(
            api_key=st.secrets.qwen.api_key,
            model='qwen-vl-plus',
            messages=messages,
            stream=True,
            incremental_output=True,
        )

        # 显示结果
        with st.expander("AI解读结果", expanded=True):
            output_placeholder = st.empty()
            full_text = ""  # 用于累加增量文本

            for chunk in response:
                try:
                    # 检查chunk是否为空
                    if not chunk:
                        continue

                    # 检查响应状态码
                    if chunk.get('status_code') != 200:
                        error_msg = chunk.get('message', chunk.get('output', {}).get('text', '未知错误'))
                        st.error(f"API请求失败: {error_msg}")
                        break

                    # 正确层级提取choices：在output字段下
                    output = chunk.get("output", {})
                    choices = output.get("choices", [])
                    if not choices:
                        continue

                    message = choices[0].get("message", {})
                    content_list = message.get("content", [])

                    # 提取文本内容并累加
                    if content_list and isinstance(content_list, list):
                        for item in content_list:
                            if "text" in item:
                                full_text += item["text"]
                                output_placeholder.markdown(full_text)

                except Exception as e:
                    st.warning(f"响应数据处理异常: {e}")
                    continue

    except Exception as e:
        st.error(f"图表解读失败: {e}")
        return


# 示例使用
if __name__ == "__main__":
    st.set_page_config(page_title="云南天气数据分析", layout="wide")

    ai_toggle()

    image_path = r"G:\大三\实习\hxf-weather-analysis\static\昆明市全部年全部月-昆明市全部年全部月-气温变化趋势.png"

    is_valid, message = validate_image_path(image_path)
    if is_valid:
        # 先在页面上显示原图（可选，方便对照）
        st.image(image_path, caption="气温变化趋势图", use_column_width=True)

        if st.session_state.is_ai:
            chart_interpretation(image_path)
    else:
        st.error(message)