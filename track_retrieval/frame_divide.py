import cv2
from utils.config import get_config
import json
import logging  # 添加导入
import subprocess
import os


main_opt = get_config()


def frame_to_video(image_name_list, interval_weight, len_weight):
    """
    Divide frames

    Args:
        image_name_list: List of image name
        interval_weight: Distinguish intervals / seconds
    Returns:
        video_lists: List of video
    """
    video_id_list = []  # list of video id
    frame_list = []  # list of frame seconds
    for image_name in image_name_list:
        video_id_list.append(image_name.split('_')[0])
        frame_list.append(image_name.split('_')[1])
    video_id_set = list(set(video_id_list))  # list id set
    frame_lists = []
    for video_id in video_id_set:
        asdf = [i for i, x in enumerate(video_id_list) if x == video_id]
        frame_lists.append([frame_list[x] for x in asdf])

    output_video = []  # output list of video
    for frame_list in frame_lists:
        frame_list = sorted(list(map(int, frame_list)))

        # Calculate video frame list
        video_lists = []
        video_list = []
        prior_frame = frame_list.pop(0)
        video_list.append(prior_frame)
        for posterior_frame in frame_list:
            if posterior_frame - prior_frame <= interval_weight:
                video_list.append(posterior_frame)
            else:
                if len(video_list) > 2 and video_list[len(video_list) - 1] - video_list[0] > len_weight:
                    video_lists.append(video_list)
                video_list = []
                video_list.append(posterior_frame)
            prior_frame = posterior_frame
        if len(video_list) > 2 and video_list[len(video_list) - 1] - video_list[0] > len_weight:
            video_lists.append(video_list)
        output_video.append(video_lists)

    return video_id_set, output_video


def trans_seconds_to_time(seconds):
    """
    Transfer seconds to time (hh:mm:ss)

    Args:
        seconds: seconds
    Returns:
        string of time
    """
    s = seconds % 60
    m = int((seconds - s) / 60) % 60
    h = int((seconds - m * 60 - s) / 3600)
    return str(h).zfill(2) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)

FFMPEG_BIN = os.path.expanduser("~/ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg")

def cut_video(start_time, end_time, video_path, save_path):
    """
    使用 FFmpeg 裁剪视频片段，输出为 H.264 编码的 MP4（兼容 Streamlit）
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 构建命令
    cmd = [
        FFMPEG_BIN,
        "-y",                          # 覆盖输出
        "-ss", str(start_time),        # 开始时间（秒，支持小数）
        "-to", str(end_time),          # 结束时间（秒）
        "-i", video_path,
        "-c:v", "libx264",             # H.264 视频编码
        "-pix_fmt", "yuv420p",         # 兼容 Safari / 所有浏览器
        "-preset", "fast",             # 编码速度 vs 压缩率
        "-crf", "23",                  # 视频质量（18~28，越小越好）
        "-an",                         # 去掉音频（如果你的视频无音）
        save_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"✅ 视频裁剪成功: {save_path}")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr
        print(f"❌ FFmpeg 错误:\n{error_msg}")
        raise RuntimeError(f"视频裁剪失败，请检查输入路径或时间范围。")
    except FileNotFoundError:
        raise RuntimeError(f"FFmpeg 未找到: {FFMPEG_BIN}")


def reorganize_by_score(frame_index, bbox, score):
    """
    Reorganize the frame_index and bbox by score (Remove duplicates)
    """
    unique_set = set(frame_index)  # change list to set, and unique data
    repeat_lst = []  # repeat data
    del_index = []  # del_index of the frame_index and bbox
    for i in unique_set:
        # if duplicates
        if frame_index.count(i) > 1:
            repeat_lst.append(i)
            # get the index of duplicates
            indexs = [index for index, value in enumerate(frame_index) if value == i]
            scores = [score[index] for index in indexs]
            # save the del_index by score
            for i, sco in enumerate(scores):
                if sco < max(scores):
                    del_index.append(indexs[i])
    # delete duplicates
    del_index.sort(reverse=True)
    for i in del_index:
        frame_index.pop(i)
        bbox.pop(i)
    return frame_index, bbox


def save_bbox_json(frame_index, bbox, score):
    """
    Save frame_index: bbox data as json

    Args:
        frame_index: index of frame (list)
        bbox: bbox of the target
        score: score of similarity between the target and the frame people
    """
    logger = logging.getLogger('run')
    if not frame_index:
        logger.info("No frames to save")
        return

    # 移除多余索引
    for i in range(len(frame_index)):
        frame_index[i] = frame_index[i].split('_')[0] + '_' + frame_index[i].split('_')[1]

    # 按分数去重
    frame_index, bbox = reorganize_by_score(frame_index, bbox, score)

    # 分割视频 ID 和帧秒数
    video_id_list = []
    frame_list = []
    for image_name in frame_index:
        video_id_list.append(image_name.split('_')[0])
        frame_list.append(image_name.split('_')[1])
    video_id_set = list(set(video_id_list))  # 唯一视频 ID 集合

    # 按视频 ID 分割帧和边界框
    frame_lists = []
    bbox_lists = []
    for video_id in video_id_set:
        asdf = [i for i, x in enumerate(video_id_list) if x == video_id]
        frame_lists.append([frame_list[x] for x in asdf])
        bbox_lists.append([bbox[x] for x in asdf])

    # 保存 JSON 文件
    for i, video_id in enumerate(video_id_set):
        zip_frame_bbox = zip(frame_lists[i], bbox_lists[i])
        sorted_zip = sorted(zip_frame_bbox, key=lambda x: int(x[0]))
        dic = dict(sorted_zip)
        filename = main_opt.json_path + video_id + '.json'
        with open(filename, 'w') as file_obj:
            json.dump(dic, file_obj)
        logger.info(f"Saved JSON file: {filename}")


if __name__ == '__main__':
    image_name_list = ['1_242_1_0', '0_242_1_0', '0_231_1_2',
                       '1_160_1_3', '1_179_1_0', '1_191_1_0',
                       '1_161_1_0', '0_159_1_0', '1_159_1_0',
                       '1_134_1_1', '1_178_1_0', '1_180_1_3',
                       '0_180_1_3', '1_177_1_0', '0_171_1_0',
                       '1_176_1_2', '0_176_1_2', '1_175_1_0',
                       '0_208_1_2', '1_208_1_2', '1_193_1_1',
                       '0_202_1_0', '1_202_1_0', '0_188_1_0',
                       '1_188_1_0', '1_220_1_1', '0_220_1_1',
                       '1_217_1_3', '0_217_1_3', '1_184_1_0',
                       '0_184_1_0', '0_236_1_1', '1_236_1_1',
                       '1_201_1_1', '0_201_1_1', '1_209_1_0',
                       '0_209_1_0', '1_230_1_0', '0_230_1_0',
                       '1_212_1_0', '0_212_1_0', '0_219_1_2',
                       '1_219_1_2', '1_199_1_0', '0_199_1_0',
                       '0_223_1_1', '1_223_1_1', '1_240_1_0',
                       '0_240_1_0', '1_226_1_0']
    interval_weight = 5
    len_weight = 5
    video_id_set, output_video = frame_to_video(image_name_list, interval_weight, len_weight)
    index = 0
    for video_lists in output_video:
        print('视频：', video_id_set[index])
        index = index + 1
        for video_list in video_lists:
            print('相似时间片段：', trans_seconds_to_time(video_list[0]), '-',
                  trans_seconds_to_time(video_list[len(video_list) - 1]))