import os
import torch
import torch.nn.functional as F
from PIL import Image
import json
from misc.utils import parse_config, setup_logger
from model.tbps_model import clip_vitb
from misc.build import load_checkpoint
from text_utils.tokenizer import tokenize
import logging
from torchvision import transforms

class CustomCUHKPEDESDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, transform, image_root):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root

        self.image = []
        self.text = []
        self.txt2img = {}
        self.img2txt = {}
        self.pid2txt = {}
        self.pid2img = {}
        self.txt_ids = []
        self.img_ids = []

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['file_path'])
            person_id = ann['id']
            if person_id not in self.pid2txt:
                self.pid2txt[person_id] = []
                self.pid2img[person_id] = []
            self.pid2img[person_id].append(img_id)
            self.img_ids.append(person_id)
            for caption in ann['captions']:
                self.text.append(caption)
                self.pid2txt[person_id].append(txt_id)
                self.txt_ids.append(person_id)
                txt_id += 1

        for tid in range(len(self.text)):
            self.txt2img[tid] = self.pid2img[self.txt_ids[tid]]
        for iid in range(len(self.image)):
            self.img2txt[iid] = self.pid2txt[self.img_ids[iid]]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

def evaluate_retrieval(config_path='config/config_orig.yaml', max_length=77):
    """按顺序检索 test_reid.json 中的每条文本，计算准确率"""
    # 解析配置
    config = parse_config(config_path)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置日志
    output_dir = os.path.join(config.logger.output_dir, config.logger.dataset_name)
    logger = setup_logger('TBPS_Retrieval', save_dir=output_dir, if_train=False)

    # 加载模型
    model = clip_vitb(config, num_classes=0)
    model.to(config.device)
    model, load_result = load_checkpoint(model, config)
    model.eval()
    logger.info(f"Loaded checkpoint: {load_result}")

    # 加载测试数据
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize((config.experiment.input_resolution, config.experiment.input_resolution),
                         interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = CustomCUHKPEDESDataset(
        ann_file=os.path.join(config.anno_dir, 'test_reid.json'),
        transform=val_transform,
        image_root=config.image_dir
    )

    # 预编码所有测试图像
    logger.info("Encoding all test images...")
    image_feats = []
    image_paths = test_dataset.image
    for image_path in image_paths:
        image = Image.open(os.path.join(config.image_dir, image_path)).convert('RGB')
        image = val_transform(image).unsqueeze(0).to(config.device)
        with torch.no_grad():
            image_feat = F.normalize(model.encode_image(image), dim=-1)
        image_feats.append(image_feat)
    image_feats = torch.cat(image_feats, dim=0)

    # 逐条检索
    correct = 0
    total = len(test_dataset.text)
    logger.info(f"Total number of test texts: {total}")

    for tid, text in enumerate(test_dataset.text):
        # 编码文本
        text_input = tokenize([text], context_length=max_length).to(config.device)
        with torch.no_grad():
            text_feat = F.normalize(model.encode_text(text_input), dim=-1)

        # 计算相似度
        sims = text_feat @ image_feats.t()
        best_idx = torch.argmax(sims, dim=-1).item()
        best_image_path = os.path.join(config.image_dir, image_paths[best_idx])
        best_score = sims[0, best_idx].item()

        # 判断检索是否正确
        expected_pids = test_dataset.txt_ids[tid]  # 文本对应的 pid
        retrieved_pid = test_dataset.img_ids[best_idx]  # 检索到的图像对应的 pid
        is_correct = (expected_pids == retrieved_pid)
        if is_correct:
            correct += 1

        # 记录检索结果
        logger.info(f"Text {tid}: {text}")
        logger.info(f"Top-1 Image: {best_image_path}, Score: {best_score:.4f}, Correct: {is_correct}")

    # 计算并记录准确率
    accuracy = correct / total
    logger.info(f"Retrieval Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Retrieval Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy

if __name__ == '__main__':
    accuracy = evaluate_retrieval()
    print(f"Final Retrieval Accuracy: {accuracy:.4f}")