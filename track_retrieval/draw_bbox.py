import cv2
from utils.config import get_config
import json
import numpy as np

main_opt = get_config()

def draw(img,left,right,color):
    #img=cv2.imread(spath)
    img=cv2.rectangle(img,left,right,color,3)
    return img

def draw_bbox(video_id, frame_path, bbox, root_path, save_path):
    img_path = str(video_id) + '_' + frame_path
    img = cv2.imread(root_path + '/' + img_path + '.jpg')
    img = draw(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0))
    cv2.imwrite(save_path + img_path + '.jpg', img)

if __name__ == '__main__':
    video_id = 1  # your video id

    # load json data
    filename = main_opt.json_path + str(video_id) + '.json'
    with open(filename) as file_obj:
      datas = json.load(file_obj)
    # print(datas)

    # divide to frame path and bbox
    # e.g. frame_paths = ['12', '13', '14'] bbox = [[390, 413, 854, 2035], [818, 298, 634, 1699], [1027, 281, 448, 1262]]
    frame_paths = list(datas.keys())
    bboxs = list(datas.values())
    # print(frame_paths)
    # print(bboxs)

    for frame_path, bbox in zip(frame_paths, bboxs):
        draw_bbox(video_id, frame_path, bbox, main_opt.frame_path, main_opt.bboxframe_path)

