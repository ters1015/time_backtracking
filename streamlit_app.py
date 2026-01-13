import streamlit as st
import os
import time
import pandas as pd
from PIL import Image

# 1. 页面基础配置
st.set_page_config(
    page_title="时光回溯系统",
    page_icon="🔍",
    layout="wide"
)

# 自定义CSS: 让按钮变大，居中，美化表格
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 20px;
        font-weight: bold;
    }
    .main_header {
        text-align: center;
        color: #1890ff;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 头部 UI
st.title("🔍时光回溯系统-基于文本和图像的视频检索演示系统")
st.header("💡 平台功能概述")
st.markdown(
    """
    <div style="background-color: #e6f7ff; border-left: 4px solid #1890ff; padding: 1rem; border-radius: 6px; margin: 1rem 0; font-size: 1.5rem; line-height: 1.6;">
        <strong>本系统是一个支持跨模态检索的视频帧检索演示平台。</strong><br><br>
        • <strong>智能检索：</strong> 依托先进的跨模态表示学习技术，系统可在<strong>多视角监控视频</strong>中快速、准确地定位与输入内容最相关的场景与目标。<br>
        • <strong>多样输入：</strong> 支持以<strong>自然语言描述</strong>或<strong>目标图像</strong>作为检索条件，实现语义与视觉的深度融合，让检索更加直观灵活。<br>
        • <strong>丰富输出：</strong> 除了展示最相关的视频帧、出现时间及匹配置信度外，系统还能自动生成并播放对应时间范围的<strong>视频片段</strong>、在楼层中的<strong>行动轨迹</strong>，便于快速回溯与验证。
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================
# 3. 资源配置 & 数据准备
# ==========================================
IMAGE_DIR = "images"

# --- 资源路径 ---
IMG_QUERY_PIC = os.path.join(IMAGE_DIR, "target9.jpg")  # 图像检索-输入图
IMG_TRAJ_PIC = os.path.join(IMAGE_DIR, "demo_traj_img.png")  # 图像检索-轨迹
IMG_VIDEO = os.path.join(IMAGE_DIR, "demo_video_img.mp4")  # 图像检索-视频

TXT_RESULT_PIC = os.path.join(IMAGE_DIR, "1_8_6_0.jpg")  # 文本检索-最佳匹配帧
TXT_TRAJ_PIC = os.path.join(IMAGE_DIR, "demo_traj_text.png")  # 文本检索-轨迹
TXT_VIDEO = os.path.join(IMAGE_DIR, "demo_video_text.mp4")  # 文本检索-视频

# --- 数据表 ---
data_img_search = [
    {"图像文件": "1_1_6_0.jpg", "时间点": "0:01", "置信度": 0.759, "距离 (Dist)": "12.22 m"},
    {"图像文件": "1_2_6_0.jpg", "时间点": "0:02", "置信度": 0.794, "距离 (Dist)": "11.78 m"},
    {"图像文件": "1_3_6_0.jpg", "时间点": "0:03", "置信度": 0.756, "距离 (Dist)": "10.21 m"},
    {"图像文件": "1_4_6_0.jpg", "时间点": "0:04", "置信度": 0.786, "距离 (Dist)": "9.87 m"},
    {"图像文件": "1_5_6_0.jpg", "时间点": "0:05", "置信度": 0.847, "距离 (Dist)": "8.73 m"},
    {"图像文件": "1_6_6_0.jpg", "时间点": "0:06", "置信度": 0.912, "距离 (Dist)": "7.29 m"},
    {"图像文件": "1_8_6_0.jpg", "时间点": "0:08", "置信度": 0.824, "距离 (Dist)": "4.65 m"},
    {"图像文件": "1_9_6_0.jpg", "时间点": "0:09", "置信度": 0.919, "距离 (Dist)": "3.30 m"},
    {"图像文件": "1_10_6_0.jpg", "时间点": "0:10", "置信度": 0.787, "距离 (Dist)": "2.16 m"},
]

data_text_search = [
    {"图像文件": "1_1_6_1.jpg", "时间点": "0:01", "置信度": 0.353, "距离 (Dist)": "12.22 m"},
    {"图像文件": "1_1_6_0.jpg", "时间点": "0:01", "置信度": 0.337, "距离 (Dist)": "12.22 m"},
    {"图像文件": "1_2_6_0.jpg", "时间点": "0:02", "置信度": 0.330, "距离 (Dist)": "11.78 m"},
    {"图像文件": "1_2_6_1.jpg", "时间点": "0:02", "置信度": 0.259, "距离 (Dist)": "11.78 m"},
    {"图像文件": "1_3_6_0.jpg", "时间点": "0:03", "置信度": 0.349, "距离 (Dist)": "10.21 m"},
    {"图像文件": "1_4_6_0.jpg", "时间点": "0:04", "置信度": 0.345, "距离 (Dist)": "9.87 m"},
    {"图像文件": "1_5_6_0.jpg", "时间点": "0:05", "置信度": 0.352, "距离 (Dist)": "8.73 m"},
    {"图像文件": "1_6_6_0.jpg", "时间点": "0:06", "置信度": 0.351, "距离 (Dist)": "7.29 m"},
    {"图像文件": "1_7_6_0.jpg", "时间点": "0:07", "置信度": 0.341, "距离 (Dist)": "5.83 m"},
    {"图像文件": "1_8_6_0.jpg", "时间点": "0:08", "置信度": 0.370, "距离 (Dist)": "4.65 m"},
    {"图像文件": "1_9_6_0.jpg", "时间点": "0:09", "置信度": 0.351, "距离 (Dist)": "3.30 m"},
    {"图像文件": "1_10_6_0.jpg", "时间点": "0:10", "置信度": 0.347, "距离 (Dist)": "2.16 m"},
]

st.subheader("📹 原始监控视频流 (Source Video)")
if os.path.exists(RAW_VIDEO_PATH):
    st.video(RAW_VIDEO_PATH)
    st.caption("原始输入视频流")
else:
    st.error(f"原始视频文件未找到，请确认已上传: {RAW_VIDEO_PATH}")

st.markdown("---")
# ==========================================
# 4. 核心逻辑控制
# ==========================================

# 初始化 Session State
if 'mode' not in st.session_state:
    st.session_state['mode'] = None

# 创建二分形式的按钮
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🖼️ 图像检索", type="primary" if st.session_state['mode'] == 'img' else "secondary"):
        st.session_state['mode'] = 'img'

with col_btn2:
    if st.button("📝 文本检索", type="primary" if st.session_state['mode'] == 'text' else "secondary"):
        st.session_state['mode'] = 'text'

# ==========================================
# 5. 结果展示逻辑
# ==========================================

if st.session_state['mode'] == 'img':
    st.markdown("### 🔹 图像检索")

    # 模拟加载
    with st.spinner():
        time.sleep(0.8)

    # 布局：Part 1 & 2 并排， Part 3 & 4 并排
    c1, c2 = st.columns([1, 2])

    # Part 1: 输入图像
    with c1:
        st.subheader("📸  检索目标")
        if os.path.exists(IMG_QUERY_PIC):
            st.image(IMG_QUERY_PIC, use_container_width=True)
        else:
            st.error(f"图片丢失: {IMG_QUERY_PIC}")

    # Part 2: 结果表格
    with c2:
        st.subheader("📊  检索结果")
        df = pd.DataFrame(data_img_search)
        st.dataframe(df.style.highlight_max(axis=0, subset=['置信度'], color='#d1ecf1'), height=300,
                     use_container_width=True)

    st.markdown("---")

    c3, c4 = st.columns(2)
    # Part 3: 轨迹
    with c3:
        st.subheader("🗺️ 目标轨迹")
        if os.path.exists(IMG_TRAJ_PIC):
            st.image(IMG_TRAJ_PIC, use_container_width=True)
        else:
            st.warning("轨迹图未找到")

    # Part 4: 视频
    with c4:
        st.subheader("🎬 视频片段")
        if os.path.exists(IMG_VIDEO):
            st.video(IMG_VIDEO)
        else:
            st.warning("视频文件未找到")


elif st.session_state['mode'] == 'text':
    st.markdown("### 🔹 文本检索")

    # 显示输入的描述文本
    st.info("📝 **输入描述**：一个戴眼镜的年轻男子，身穿白色上衣和黑色长裤")

    # 模拟加载
    with st.spinner():
        time.sleep(0.8)

    c1, c2 = st.columns([1, 2])

    # Part 1: 最佳匹配帧 (因为是文本检索，所以展示系统找到的最好的那张图)
    with c1:
        st.subheader("📸 最佳匹配结果")
        if os.path.exists(TXT_RESULT_PIC):
            st.image(TXT_RESULT_PIC, caption="Top-1 Match: 1_8_6_0.jpg", use_container_width=True)
        else:
            st.error(f"图片丢失: {TXT_RESULT_PIC}")

    # Part 2: 结果表格
    with c2:
        st.subheader("📊 检索结果")
        df = pd.DataFrame(data_text_search)
        st.dataframe(df.style.highlight_max(axis=0, subset=['置信度'], color='#fff3cd'), height=300,
                     use_container_width=True)

    st.markdown("---")

    c3, c4 = st.columns(2)
    # Part 3: 轨迹
    with c3:
        st.subheader("🗺️ 目标轨迹")
        if os.path.exists(TXT_TRAJ_PIC):
            st.image(TXT_TRAJ_PIC, use_container_width=True)
        else:
            st.warning("轨迹图未找到")

    # Part 4: 视频
    with c4:
        st.subheader("🎬 视频片段")
        if os.path.exists(TXT_VIDEO):
            st.video(TXT_VIDEO)
        else:
            st.warning("视频文件未找到")

else:
    # 默认提示
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #666;">
            <h3>👈 请点击上方按钮开始演示</h3>
        </div>
    """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>© 2026 TBPS System | Time Backtracking Pedestrian Search</div>",
    unsafe_allow_html=True)