from PIL import Image
import os
import torch
from torchvision import transforms
from run.people.defaults import get_default_cfg
from run.people.models.seqnet import SeqNet
from utils.config import get_config
# --- 新增 ---
# 1. 导入 OSNet (假设 osnet.py 已放入 run/people/models/)
from run.people.models.osnet import osnet_ibn_x1_0
# --- 结束 ---

opt = get_config()
use_cuda = torch.cuda.is_available()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


def extract_feature_people(image):
    # cfg = get_default_cfg() # (不再需要)
    # cfg.merge_from_file(opt.cfg_path_people) # (不再需要)
    # cfg.freeze() # (不再需要)

    # --- !! 关键修改：加载 OSNet !! ---
    ckpt_path_osnet = opt.model_path_osnet  # (使用 config.py 中新加的路径)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #

    model = osnet_ibn_x1_0(num_classes=751, loss='softmax', pretrained=False)  # 假设OSNet的初始化方式

    # *** 注意：加载OSNet权重的方式必须与步骤3中一致 ***
    try:
        osnet_weights = torch.load(ckpt_path_osnet, map_location=device)
        if 'model' in osnet_weights:
            model.load_state_dict(osnet_weights['model'], strict=False)
        elif 'state_dict' in osnet_weights:
            model.load_state_dict(osnet_weights['state_dict'], strict=False)
        else:
            model.load_state_dict(osnet_weights, strict=False)
    except Exception as e:
        print(f"Error loading OSNet weights: {e}. Please check the checkpoint file structure.")
        return []  # 加载失败则退出

    model.to(device)
    # resume_from_ckpt(ckpt_path, model) # (不再需要)
    model.eval()  #

    # 2. 修改预处理
    # 尺寸 (256, 128) 和归一化参数与步骤3中的 transform_osnet 保持一致
    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode == 'RGBA' else x),  #
        transforms.Resize((256, 128)),  # (保持或改为 OSNet 尺寸)
        transforms.ToTensor(),  #
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #
    ])

    img_new = transform_test(image).unsqueeze(0)  #
    image_new = img_new.to(device)  #

    # 3. 修改推理逻辑
    # output = model(image_new) # (旧逻辑)
    feature_tensor = model(image_new)

    # L2 归一化 (必须与预处理阶段一致)
    feature_tensor = torch.nn.functional.normalize(feature_tensor, p=2, dim=1)

    feature = feature_tensor.cpu().detach().numpy().flatten().tolist()

    # 4. 修改返回逻辑
    # (删除所有解析 output[0]['embeddings'] 和 'scores' 的逻辑)

    if not feature or len(feature) == 0:
        print("Warning: OSNet embeddings is empty")
        return []

    print(f"Extracted OSNet feature dimension: {len(feature)}")
    return feature

    # --- 旧逻辑删除 ---
    # print(f"Raw output: {output}")  # 调试
    # if isinstance(output, list) and len(output) > 0 and 'embeddings' in output[0]:
    #     ...
    # else:
    #     print("Error: output structure mismatch")
    #     return []
    # --- 结束 ---

def resume_from_ckpt(ckpt_path, model, optimizer=None, lr_scheduler=None):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    return ckpt["epoch"]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    path = '/home/zhangzhewei/时间回溯/items/photo_test/target15.png'
    img = Image.open(path)
    # 显示输入图片，确认是行人
    plt.imshow(img)
    plt.title("Input Image (Pedestrian)")
    plt.show()

    feature = extract_feature_people(img)
    print(f"特征向量是否为空: {not feature}")
    if feature:
        print(f"特征向量长度: {len(feature)}")
        print(f"特征向量前5值: {feature[:5]}")