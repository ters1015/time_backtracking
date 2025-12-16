from yacs.config import CfgNode as CN

_C = CN()

# socket communication
_C.ip = '10.112.44.157'
_C.transport_port = 8888
_C.run_port = 8899

# data path
# _C.video_path = '/home/lab239-5/users/ZTE/liuzhe_items/video_test'
_C.video_path = '/home/zhangzhewei/time_backtracking/items/video_test_short'
_C.data_root = '/home/zhangzhewei/毕设/TBPS'
# _C.video_path = '/mnt/DataDrive5/zte/preprocess/video_test'
# save path of key frame
# _C.frame_path = '/home/lab239-5/users/ZTE/liuzhe_items/extract_test'
_C.frame_path = '/home/zhangzhewei/time_backtracking/items/extract_test_short'
# _C.frame_path = '/mnt/DataDrive5/zte/preprocess/extract_test'
_C.pedestrian_save_path = '/home/zhangzhewei/time_backtracking/items/pedestrian_images'
# save bbox frame picture
_C.bboxframe_path = '/home/zhangzhewei/time_backtracking/items/extract_result'
# save path of bbox json
_C.json_path = '/home/zhangzhewei/time_backtracking/items/bbox_result'

# model path

_C.model_path_people = '/home/zhangzhewei/time_backtracking/fusion_project/epoch_19.pth'
_C.cfg_path_people = '/home/zhangzhewei/time_backtracking/fusion_project/config.yaml'
# 添加 OSNet 模型的权重路径
_C.model_path_osnet = '/home/zhangzhewei/time_backtracking/fusion_project/osnet_ibn_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'  # <--- 修改为您 OSNet 权重的实际路径
_C.gpu = '0'

# 运行与处理文件时，首先指定是否要删除索引
# 若是，则指定删除的索引类型，然后指定电影id, delete_multi_search为[]时，不删除
# _C.delete_multi_search = ['face', 'people', 'pedestrian']
_C.delete_multi_search = []
# 范围删除，指定要删除电影id的上下界，为空时全部删除
_C.delete_film = []
# 指定视频结构化预处理的类型
# _C.deal_type = ['face', 'people', 'pedestrian']
# _C.deal_type = ['face']
# 指定分帧方式，value from ['keyframe,'sec']
_C.ext_frame = 'sec'
# 如果按秒分帧，需要 指定采样频率,默认为一秒取一帧
_C.sec_frame_fq = 1
# 是否后台检索时，是否设定相似度阈值
_C.threshold_bool = True


def update_config(config, opt):
    config.defrost()
    config.freeze()


def get_config(opt=None):
    config = _C.clone()
    if opt:
        update_config(config, opt)
    return config
