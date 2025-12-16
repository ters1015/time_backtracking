import streamlit as st
import os
import subprocess
import torch
import json
import re
import requests
import hashlib
import cv2
import numpy as np
import tempfile
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.write("🔄 正在尝试导入自定义模块...")

from text_utils.tokenizer import tokenize
st.write("✅ text_utils 导入成功")

from misc.build import load_checkpoint
from misc.utils import parse_config
st.write("✅ misc 导入成功")

from model.tbps_model import clip_vitb
st.write("✅ model 导入成功")

from run.people.ext_feature_people_new import extract_feature_people
st.write("✅ run.people 导入成功")

from utils.search_utils import search
st.write("✅ utils 导入成功")

from track_retrieval.frame_divide import trans_seconds_to_time, cut_video
st.write("✅ track_retrieval 导入成功")

# ================= 1. 路径与环境配置 (云端适配核心) =================

# 获取当前文件所在目录作为根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义各资源文件夹的绝对路径 (基于 ROOT_DIR)
ITEMS_DIR = os.path.join(ROOT_DIR, "items")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

# 具体文件路径
CALIB_PATH = os.path.join(ROOT_DIR, "calibration.json")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8m_200e.pt")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config_orig.yaml")

# 数据资源路径
EXTRACTED_FRAMES_DIR = os.path.join(ITEMS_DIR, "extract_test_short")
TARGET_VIDEO_PATH1 = os.path.join(ITEMS_DIR, "video_test_short", "1.mp4")
QUERY_IMAGE_DIR = os.path.join(ITEMS_DIR, "photo_test")
MAP_IMAGE_PATH = os.path.join(ITEMS_DIR, "北邮教三6楼2-F1.jpg")

# 百度翻译 API (优先从 Secrets 读取，防止 Key 泄露)
try:
    BAIDU_APP_ID = st.secrets["baidu"]["app_id"]
    BAIDU_SECRET_KEY = st.secrets["baidu"]["secret_key"]
except:
    BAIDU_APP_ID = ""
    BAIDU_SECRET_KEY = ""
    # st.sidebar.warning("⚠️ 未检测到百度翻译密钥，文本检索功能可能受限。")

BAIDU_API_URL = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
TO_LANG = 'en'
FROM_LANG = 'auto'

# ================= 2. 参数配置 =================

# 人体比例参数
BODY_HEIGHT_MM = 1720.1
GLABELLA_TO_VERTEX_MM = 75
FACE_HEIGHT_MM = 200
k_foot = (BODY_HEIGHT_MM - GLABELLA_TO_VERTEX_MM) / FACE_HEIGHT_MM

# 地图与摄像头参数 (您提供的数据)
CAM_PIXEL_X = 2915
CAM_PIXEL_Y = 1256
PIXELS_PER_METER = 3629 / 71.60

PRESET_IMAGES = ["target9.jpg"]

VIDEO_WORLD_START_TIME = {
    "1": datetime(2025, 11, 21, 14, 6, 0),
    "2": datetime(2025, 11, 21, 14, 6, 0),
    "6": datetime(2025, 11, 21, 14, 6, 0),
}

# ================= 3. 初始化加载 =================
st.set_page_config(
    page_title="🔍时光回溯系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载标定数据
if os.path.exists(CALIB_PATH):
    with open(CALIB_PATH) as f:
        CALIB_DATA = json.load(f)
else:
    CALIB_DATA = None
    st.warning(f"⚠️ 标定文件未找到: {CALIB_PATH}")


# 加载 YOLO Face 模型
@st.cache_resource
def load_face_model():
    if not os.path.exists(FACE_MODEL_PATH):
        st.error(f"❌ 模型文件未找到: {FACE_MODEL_PATH}\n请确保模型已上传至 models 文件夹。")
        return None
    try:
        return YOLO(FACE_MODEL_PATH)
    except Exception as e:
        st.error(f"YOLO 模型加载失败: {e}")
        return None


face_model = load_face_model()


# ================= 4. 核心功能函数 =================

def draw_trajectory_on_map(trajectory_points, map_path):
    """
    在楼层平面图上绘制摄像头位置和行人轨迹
    trajectory_points: [{'dist': 米数, 'time': 时间字符串}, ...]
    """
    if not os.path.exists(map_path):
        st.error(f"地图文件不存在: {map_path}")
        return

    try:
        pil_img = Image.open(map_path)
        img_w, img_h = pil_img.size
    except Exception as e:
        st.error(f"无法读取地图图片: {e}")
        return

    plot_x = []
    plot_y = []
    plot_text = []

    for item in trajectory_points:
        dist_m = item['dist']
        time_str = item['time']

        # 映射逻辑：摄像头向左看 (X轴负方向)
        px = CAM_PIXEL_X - (dist_m * PIXELS_PER_METER)
        py = CAM_PIXEL_Y

        # 简单的边界检查
        if 0 <= px <= img_w:
            plot_x.append(px)
            plot_y.append(py)
            plot_text.append(f"时间: {time_str}<br>距离: {dist_m:.2f}m")

    fig = go.Figure()

    # 添加背景图
    fig.add_layout_image(
        dict(
            source=pil_img,
            xref="x", yref="y",
            x=0, y=img_h,
            sizex=img_w, sizey=img_h,
            sizing="stretch",
            opacity=1.0,
            layer="below"
        )
    )

    # 绘制行人轨迹
    if plot_x:
        fig.add_trace(go.Scatter(
            x=plot_x, y=plot_y,
            mode='lines+markers',
            name='行人轨迹',
            line=dict(color='#1E90FF', width=4),
            marker=dict(size=10, color='cyan', line=dict(width=1, color='white')),
            text=plot_text,
            hoverinfo='text'
        ))
        # 起点
        fig.add_trace(go.Scatter(
            x=[plot_x[0]], y=[plot_y[0]],
            mode='markers', name='起点',
            marker=dict(size=12, color='green', symbol='circle')
        ))
        # 终点
        fig.add_trace(go.Scatter(
            x=[plot_x[-1]], y=[plot_y[-1]],
            mode='markers', name='最新位置',
            marker=dict(size=12, color='orange', symbol='star')
        ))

    # 绘制摄像头
    fig.add_trace(go.Scatter(
        x=[CAM_PIXEL_X], y=[CAM_PIXEL_Y],
        mode='markers+text',
        name='摄像头',
        marker=dict(size=17, color='red', symbol='triangle-left'),
        text=["摄像头"], textposition="top center",
        textfont=dict(size=14, color="red", family="Arial Black"),
        hoverinfo='text'
    ))

    fig.update_layout(
        width=900, height=600,
        xaxis=dict(range=[0, img_w], showgrid=False, visible=False),
        yaxis=dict(range=[0, img_h], showgrid=False, visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        dragmode='pan'
    )
    st.plotly_chart(fig, use_container_width=True)


def calculate_z_distance(vb):
    if CALIB_DATA is None: return -1
    f, yc, vc, v0 = CALIB_DATA['f'], CALIB_DATA['yc'], CALIB_DATA['vc'], CALIB_DATA['v0']
    numerator = yc * (f ** 2) - yc * (vb - vc) * (vc - v0)
    denominator = f * (vb - v0)
    if abs(denominator) < 1e-6: return -1
    z = numerator / denominator
    return z / 1000.0


def estimate_distance_for_target(video_id, seconds, query_feat_or_text, retrieval_mode, clip_model=None,
                                 clip_transform=None):
    """
    检测人脸 -> 推算脚点 -> 计算距离
    返回: 距离 (float) 或 None
    """
    frame_name = f"{video_id}_{seconds}.jpg"
    frame_path = os.path.join(EXTRACTED_FRAMES_DIR, frame_name)

    if not os.path.exists(frame_path): return None
    frame = cv2.imread(frame_path)
    if frame is None: return None

    if face_model is None: return None
    # verbose=False 减少日志
    results = face_model(frame, conf=0.35, iou=0.5, verbose=False)[0]
    if not results.boxes: return None

    best_score = -1
    target_dist = None

    # 准备 Query Feature
    query_feat = query_feat_or_text
    if retrieval_mode == "text" and clip_model and isinstance(query_feat, str):
        with torch.no_grad():
            tk = tokenize([query_feat], context_length=77).to(device)
            query_feat = torch.nn.functional.normalize(clip_model.encode_text(tk), dim=-1)

    h, w = frame.shape[:2]

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        vf, L = y1, y2 - y1
        vb = vf + k_foot * L

        # 裁剪身体进行 Re-ID
        fw = x2 - x1
        bx1, bx2 = max(0, x1 - fw), min(w, x2 + fw)
        by1, by2 = max(0, y1 - int(L * 0.5)), min(h, int(vb))
        crop = frame[by1:by2, bx1:bx2]

        if crop.size == 0: continue
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        score = 0
        if retrieval_mode == "text" and clip_model:
            with torch.no_grad():
                ifeat = torch.nn.functional.normalize(clip_model.encode_image(clip_transform(crop_pil).to(device)),
                                                      dim=-1)
                score = (query_feat @ ifeat.t()).item()
        elif retrieval_mode == "image":
            vec = extract_feature_people(crop_pil)
            if vec is not None:
                score = np.dot(query_feat, vec) / (np.linalg.norm(query_feat) * np.linalg.norm(vec) + 1e-6)

        if score > best_score:
            best_score = score
            target_dist = calculate_z_distance(vb)

    return target_dist


# ================= 5. 视频处理与辅助函数 =================

def get_video_save_path(start, end, r_type, vid, slug="", t_name=None):
    # 输出到 output_videos 临时文件夹
    out_dir = os.path.join(ROOT_DIR, 'output_videos')
    os.makedirs(out_dir, exist_ok=True)

    s, e = max(0, start - 1), end + 1
    if r_type == "text":
        name = f'{vid}_{slug}_{s}-{e}.mp4'
        save_path = os.path.join(out_dir, name)
    else:
        name = f'{vid}_{t_name or "target"}_0.mp4'
        save_path = os.path.join(out_dir, name)

    src_path = os.path.join(ITEMS_DIR, 'video_test_short', f'{vid}.mp4')
    return src_path, save_path, (s, e)


def add_timestamp_overlay(input_video, output_video, start_dt, end_dt):
    # OpenCV 绘制 + FFmpeg 转码
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened(): return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用 tempfile 防止文件冲突
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    # mp4v 编码兼容性较好用于中间处理
    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        prog = idx / count if count > 0 else 0
        if prog > 1: prog = 1

        # 背景条
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        # 进度条
        bar_w = int(w * prog)
        cv2.rectangle(frame, (0, h - 30), (bar_w, h), (255, 144, 30), -1)

        s_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        e_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')

        # 文字
        cv2.putText(frame, s_str, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tw = cv2.getTextSize(e_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(frame, e_str, (w - tw - 10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        idx += 1

    cap.release();
    out.release()

    # FFmpeg 转码为 H.264 以在浏览器播放
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_path, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", output_video],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        os.remove(temp_path)
        return True
    except:
        if os.path.exists(temp_path): os.remove(temp_path)
        return False


def get_world_time(video_id, second_offset):
    if video_id not in VIDEO_WORLD_START_TIME: return None
    return VIDEO_WORLD_START_TIME[video_id] + timedelta(seconds=second_offset)


def merge_videos_multiview(paths, output_path):
    if not paths: return
    if len(paths) == 1:
        import shutil
        shutil.copy(paths[0], output_path)
    else:
        subprocess.run(["ffmpeg", "-y", "-i", paths[0], "-i", paths[1], "-filter_complex", "hstack", "-c:v", "libx264",
                        output_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def translate_to_english(text):
    if not text or not BAIDU_APP_ID: return text
    salt = os.urandom(8).hex()
    sign = hashlib.md5((BAIDU_APP_ID + text + salt + BAIDU_SECRET_KEY).encode('utf-8')).hexdigest()
    try:
        res = requests.post(BAIDU_API_URL,
                            params={'appid': BAIDU_APP_ID, 'q': text, 'from': FROM_LANG, 'to': TO_LANG, 'salt': salt,
                                    'sign': sign}).json()
        return res.get('trans_result', [{}])[0].get('dst', text)
    except:
        return text


def extract_seconds(fn):
    m = re.match(r"(\d+)_(\d+)_", fn) or re.match(r"(\d+)_(\d+)_\d+_\d+", fn)
    return int(m.group(2)) if m else 0


def format_time(s): return f"{s // 60:01d}:{s % 60:02d}"


def group_results_by_video_id(results):
    from collections import defaultdict
    g = defaultdict(list)
    for sim, fp, sec in results:
        vid = os.path.basename(fp).split('_')[0]
        g[vid].append((sim, fp, sec))
    return g


def generate_and_display_all_cropped_videos(results, r_type, query_slug="", target_name=None):
    if not results: return

    # 1. 数据分组与排序
    grouped = group_results_by_video_id(results)
    try:
        sorted_vids = sorted(grouped.keys(), key=lambda x: int(x))
    except:
        sorted_vids = sorted(grouped.keys())

    # 2. 预计算时间范围 (用于绘制进度条)
    all_times = []
    meta = {}
    for vid in sorted_vids:
        sim_fp_secs = grouped[vid]
        secs = [item[2] for item in sim_fp_secs]  # 提取秒数
        s, e = min(secs), max(secs)
        ws, we = get_world_time(vid, s), get_world_time(vid, e)
        meta[vid] = {'s': s, 'e': e, 'ws': ws, 'we': we}
        if ws and we: all_times.append((ws, we))

    g_min = min([t[0] for t in all_times]) if all_times else datetime.now()
    g_max = max([t[1] for t in all_times]) if all_times else g_min
    total_dur = (g_max - g_min).total_seconds() or 1.0

    st.markdown("### 🎬 视频片段")
    processed_paths = []
    cards_data = []

    # 3. 处理每个视频
    for vid in sorted_vids:
        m = meta[vid]
        src, dst, times = get_video_save_path(m['s'], m['e'], r_type, vid, query_slug, target_name)

        if not os.path.exists(src): continue
        dst_over = dst.replace(".mp4", "_overlay.mp4")

        try:
            cut_video(times[0], times[1], src, dst)
            # 尝试加水印
            if m['ws'] and m['we'] and add_timestamp_overlay(dst, dst_over, m['ws'], m['we']):
                processed_paths.append(dst_over)
            else:
                processed_paths.append(dst)

            # 收集卡片数据
            if m['ws'] and m['we']:
                offset = (m['ws'] - g_min).total_seconds()
                dur = (m['we'] - m['ws']).total_seconds()
                lp = max(0, min((offset / total_dur) * 100, 100))
                wp = max(0, min((dur / total_dur) * 100, 100))
                t_str = f"{m['ws'].strftime('%Y-%m-%d %H:%M:%S')} - {m['we'].strftime('%H:%M:%S')}"
                cards_data.append({'id': vid, 'text': f"视角 {vid}: {t_str}", 'l': f"{lp:.1f}%", 'w': f"{wp:.1f}%"})
        except Exception as e:
            print(f"Video Error {vid}: {e}")
            pass

    # 4. 合并播放
    if processed_paths:
        merged = os.path.join(ROOT_DIR, 'output_videos', 'merged.mp4')
        merge_videos_multiview(processed_paths, merged)
        if os.path.exists(merged): st.video(merged, width=700)

    # 5. 显示进度条卡片
    if cards_data:
        st.markdown("### ⏱️ 时空回溯轨迹")
        html = ""
        for d in cards_data:
            vid = str(d['id'])
            if vid == "1":
                bg, bd, tr, bar = "#f9f0ff", "#d3adf7", "#ebd6ff", "#9254de"
            elif vid == "2":
                bg, bd, tr, bar = "#e6f7ff", "#91d5ff", "#bae7ff", "#1890ff"
            else:
                bg, bd, tr, bar = "#f5f5f5", "#d9d9d9", "#e8e8e8", "#8c8c8c"

            html += f"""
            <div style="flex:1; background:{bg}; border:1px solid {bd}; border-radius:5px; padding:10px; font-size:14px; color:#000; display:flex; flex-direction:column; justify-content:space-between; overflow:hidden;">
                <div style="display:flex; align-items:center; margin-bottom:8px;"><span style="margin-right:8px;">🕒</span><span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{d['text']}</span></div>
                <div style="width:100%; height:6px; background:{tr}; border-radius:3px; position:relative;"><div style="position:absolute; left:{d['l']}; width:{d['w']}; height:100%; background:{bar}; border-radius:3px; opacity:0.8;"></div></div>
            </div>"""
        st.markdown(f'<div style="width:700px; display:flex; gap:10px; margin-top:-10px;">{html}</div>',
                    unsafe_allow_html=True)


# ================= 6. 模型与数据加载 (路径动态修正) =================
@st.cache_resource
def load_clip_config():
    try:
        config = parse_config(CONFIG_PATH)
        # 强制将配置中的路径指向云端实际存在的文件夹
        config.image_dir = EXTRACTED_FRAMES_DIR

        # 云端环境通常无GPU，使用CPU
        config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = clip_vitb(config)
        model, _ = load_checkpoint(model, config)
        model = model.to(config.device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((config.experiment.input_resolution, config.experiment.input_resolution),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.experiment.input_resolution),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
            lambda x: x.unsqueeze(0) if x.dim() == 3 else x
        ])

        # 尝试加载标注文件
        anno_path = os.path.join(ITEMS_DIR, "jiaosan_pedestrian_annotations.json")
        # 如果 items 里没有，试着在根目录找
        if not os.path.exists(anno_path):
            anno_path = os.path.join(ROOT_DIR, "jiaosan_pedestrian_annotations.json")

        if os.path.exists(anno_path):
            with open(anno_path, "r") as f:
                data = json.load(f)
            valid_data = []
            for e in data:
                # 兼容处理路径
                fp = os.path.basename(
                    e["file_path"][0].strip() if isinstance(e["file_path"], list) else e["file_path"].strip())
                if not fp.lower().endswith(('.jpg', '.jpeg')): fp += '.jpg'
                if os.path.exists(os.path.join(config.image_dir, fp)):
                    valid_data.append({**e, "file_path_clean": fp})
        else:
            st.error(f"❌ 标注文件未找到: {anno_path}")
            valid_data = []

        return config, model, transform, valid_data, config.device
    except Exception as e:
        st.error(f"Config Load Error: {e}")
        return None, None, None, [], "cpu"


config, model, transform, valid_data, device = load_clip_config()
if config is None: st.stop()


@st.cache_data
def text_to_image_query(_model, q_text, v_data, _trans, thresh=-1.0):
    _model.eval()
    try:
        tk = tokenize([q_text], context_length=77).to(device)
        with torch.no_grad():
            tf = torch.nn.functional.normalize(_model.encode_text(tk), dim=-1)
            res = []
            for e in v_data:
                try:
                    img = Image.open(os.path.join(config.image_dir, e["file_path_clean"])).convert('RGB')
                    if_ = torch.nn.functional.normalize(_model.encode_image(_trans(img).to(device)), dim=-1)
                    sim = (tf @ if_.t()).item()
                    res.append((sim, e["file_path_clean"], extract_seconds(e["file_path_clean"])))
                except:
                    continue
            return sorted(res, key=lambda x: -x[0])
    except:
        return []


@st.cache_data
def image_to_image_query_osnet(img_path, thresh=0.8):
    try:
        qv = extract_feature_people(Image.open(img_path).convert('RGB'))
        if qv is None: return [], [], []
        r_str, bbox, scores = search(qv, 'pedestrian', True, thresh)
        if not r_str: return [], [], []
        res = []
        for i, fp in enumerate(r_str.split(' ')):
            res.append((scores[i] if i < len(scores) else 0, fp, extract_seconds(fp)))
        return sorted(res, key=lambda x: -x[0]), bbox, scores
    except:
        return [], [], []


# ================= 主页面 =================
st.title("🔍时光回溯系统-基于文本和图像的视频检索演示系统")
st.header("💡 平台功能概述")
st.markdown(
    """
    <div style="background-color: #e6f7ff; border-left: 4px solid #1890ff; padding: 1rem; border-radius: 6px; margin: 1rem 0; font-size: 1.5rem; line-height: 1.6;">
        <strong>本系统是一个支持跨模态检索的视频帧检索演示平台。</strong><br><br>
        • <strong>智能检索：</strong> 依托先进的跨模态表示学习技术，系统可在<strong>多视角监控视频</strong>中快速、准确地定位与输入内容最相关的场景与目标。<br>
        • <strong>多样输入：</strong> 支持以<strong>自然语言描述</strong>或<strong>目标图像</strong>作为检索条件，实现语义与视觉的深度融合，让检索更加直观灵活。<br>
        • <strong>丰富输出：</strong> 除了展示最相关的视频帧、出现时间及匹配置信度外，系统还能自动生成并播放对应时间范围的<strong>视频片段</strong>，便于快速回溯与验证。
    </div>
    """,
    unsafe_allow_html=True
)

st.header("🎥 平台功能演示")

if os.path.exists(TARGET_VIDEO_PATH1): st.video(TARGET_VIDEO_PATH1, width=700)

s_type = st.selectbox("检索类型", ["文本检索", "图像检索"])

if s_type == "文本检索":
    q_in = st.text_area("输入查询文本", height=150, placeholder="例如：一个戴眼镜的年轻男子...")
    if st.button("🔍开始文本检索", type="primary") and q_in:
        q_en = translate_to_english(q_in)

        with st.expander("🛠️ 调试信息", expanded=False):
            st.write(f"翻译结果: {q_en}")
            st.write(f"数据库图片数: {len(valid_data)}")

        with st.spinner("检索中..."):
            results = text_to_image_query(model, q_en or q_in, valid_data, transform, -1.0)

        if results:
            st.success(f"找到 {len(results)} 个结果")

            with torch.no_grad():
                q_tokens = tokenize([q_en or q_in], context_length=77).to(device)
                q_feat = torch.nn.functional.normalize(model.encode_text(q_tokens), dim=-1)

            traj_data, df_data = [], []
            display_res = sorted(results[:20], key=lambda x: x[2])
            bar = st.progress(0, text="计算距离...")

            for i, (sim, fp, sec) in enumerate(display_res):
                vid = os.path.basename(fp).split('_')[0]

                # 测距 (修复版调用，只返回 dist)
                dist = estimate_distance_for_target(vid, sec, q_feat, "text", model, transform)

                d_str = f"{dist:.2f} m" if (dist and dist > 0) else "N/A"

                if dist and dist > 0:
                    wt = get_world_time(vid, sec)
                    ts = wt.strftime('%Y-%m-%d %H:%M:%S') if wt else format_time(sec)
                    traj_data.append({'dist': dist, 'time': ts})

                df_data.append({'图像': fp, '时间': format_time(sec), '置信度': f"{sim:.3f}", '距离': d_str})
                bar.progress((i + 1) / len(display_res))

            bar.empty()
            st.dataframe(df_data, use_container_width=True)

            st.markdown("### 📍 行人轨迹追踪")
            if traj_data:
                draw_trajectory_on_map(traj_data, MAP_IMAGE_PATH)
            else:
                st.info("未检测到有效的距离数据")

            slug = "_".join(q_in.split()[:3]).replace('/', '_')
            generate_and_display_all_cropped_videos(display_res, "text", query_slug=slug)

elif s_type == "图像检索":
    sel_img = st.selectbox("选择查询图像", PRESET_IMAGES)
    if st.button("🔍开始图像检索", type="primary") and sel_img:
        q_path = os.path.join(QUERY_IMAGE_DIR, sel_img)
        st.image(q_path, width=200)

        with st.spinner("检索中..."):
            results, _, _ = image_to_image_query_osnet(q_path, 0.75)

        if results:
            st.success(f"找到 {len(results)} 个结果")
            try:
                qv = extract_feature_people(Image.open(q_path).convert('RGB'))
            except:
                qv = None

            traj_data, df_data = [], []
            display_res = sorted(results[:20], key=lambda x: x[2])
            bar = st.progress(0, text="计算距离...")

            for i, (sim, fp, sec) in enumerate(display_res):
                vid = os.path.basename(fp).split('_')[0]
                dist = estimate_distance_for_target(vid, sec, qv, "image")

                d_str = f"{dist:.2f} m" if (dist and dist > 0) else "N/A"

                if dist and dist > 0:
                    wt = get_world_time(vid, sec)
                    ts = wt.strftime('%Y-%m-%d %H:%M:%S') if wt else format_time(sec)
                    traj_data.append({'dist': dist, 'time': ts})

                df_data.append({'图像': fp, '时间': format_time(sec), '置信度': f"{sim:.3f}", '距离': d_str})
                bar.progress((i + 1) / len(display_res))

            bar.empty()
            st.dataframe(df_data, use_container_width=True)

            st.markdown("### 📍 行人轨迹追踪")
            if traj_data: draw_trajectory_on_map(traj_data, MAP_IMAGE_PATH)

            generate_and_display_all_cropped_videos(display_res, "image", target_name=os.path.splitext(sel_img)[0])