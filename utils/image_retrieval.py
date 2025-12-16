import os
import torch
from PIL import Image
import logging
from run.people.ext_feature_people_new import extract_feature_people
from utils.search_utils import search
from utils.config import get_config
from utils.logger import create_logger
import datetime

def retrieve_by_image(image_path, index_name='pedestrian', threshold=0.51):
    """
    输入图片路径，检索 Top-1 匹配的图片信息。
    :param image_path: 图片文件路径（如 '/home/zhangzhewei/时间回溯/items/photo_test/target15.png'）
    :param index_name: Elasticsearch 索引名称（默认 'pedestrian'）
    :param threshold: 相似度阈值（默认 0.51）
    :return: 字典，包含 image_name, bbox, score
    """
    # 初始化配置和日志
    opt = get_config()
    now_date = datetime.datetime.now().strftime('%Y-%m-%d')
    this_dir = os.path.split(os.path.realpath(__file__))[0]
    logger = create_logger(output_dir=os.path.join(this_dir, 'log/run'), mode=now_date, name='run')

    # 加载图片
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return {}

    # 提取特征
    start_time = datetime.datetime.now()
    vector = extract_feature_people(image)
    if not vector:
        logger.warning(f"No feature extracted for image {image_path}")
        return {}
    logger.info(f"Feature extracted for {image_path}, length: {len(vector)}")
    feature_time = (datetime.datetime.now() - start_time).total_seconds()

    # Elasticsearch 检索
    threshold_bool = opt.threshold_bool
    results, bbox, scores = search(vector, index_name, threshold_bool, threshold)
    logger.info(f"Search completed, results: {results}, bbox: {bbox}, scores: {scores}")

    # 处理检索结果
    if not results or not scores:
        logger.info("No similar images found")
        return {}

    # 获取 Top-1 结果
    image_names = results.strip().split()
    if not image_names:
        logger.info("No valid image names in results")
        return {}

    top1_result = {
        'image_name': image_names[0],
        'bbox': bbox[0] if bbox else [],
        'score': scores[0] if scores else 0.0
    }
    logger.info(f"Top-1 Image: {top1_result['image_name']}, BBox: {top1_result['bbox']}, Score: {top1_result['score']:.4f}")
    logger.info(f"Feature extraction time: {feature_time:.4f}s")

    return top1_result

if __name__ == '__main__':
    # 测试图片
    test_image_path = '/home/zhangzhewei/毕设/TBPS/imgs1/train_query/p8848_s17661.jpg'
    result = retrieve_by_image(test_image_path)
    print(f"Retrieved result: {result}")