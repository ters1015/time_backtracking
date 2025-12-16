import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFilter
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

class ps_train_dataset(Dataset):
    def __init__(self, ann_root, image_root, transform, aug_ss, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file))
        self.transform = transform

        self.person2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        self.pairs = []

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            if 'captions_bt' not in ann:
                ann['captions_bt'] = [''] * len(ann['captions'])
            for caption, caption_bt in zip(ann['captions'], ann['captions_bt']):
                caption = pre_caption(caption, max_words)
                caption_bt = pre_caption(caption_bt, max_words)
                self.pairs.append((image_path, caption, caption_bt, person_idx))
                self.person2text[person_idx].append(caption)

        self.augmentation_ss = aug_ss

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, caption_bt, person = self.pairs[index]
        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        aug1 = self.transform(image_pil.convert('RGB'))
        aug_ss_1 = self.augmentation_ss(image_pil)
        aug_ss_2 = self.augmentation_ss(image_pil)
        return {
            'image': image,
            'caption': caption,
            'caption_bt': caption_bt,
            'id': person,
            'aug1': aug1,
            'aug_ss_1': aug_ss_1,
            'aug_ss_2': aug_ss_2
        }

class ps_eval_dataset(Dataset):
    def __init__(self, ann_root, test_sample, image_root, transform, split, max_words=30):
        ann_file = ann_root
        try:
            anns = json.load(open(ann_file, 'r'))
        except Exception as e:
            raise ValueError(f"无法加载 JSON 文件 {ann_file}: {e}")
        self.transform = transform

        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []
        self.test_sample = test_sample
        texts = []
        texts_ts = []

        valid_anns = []
        for idx, ann in enumerate(anns):
            file_path = ann['file_path'][0] if isinstance(ann['file_path'], list) else ann['file_path']
            file_path = os.path.basename(file_path).strip()
            if not file_path or file_path.lower() == 'jpg' or ',' in file_path:
                continue
            if not (file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg')):
                file_path += '.jpg'
            image_path = os.path.join(image_root, file_path)
            if os.path.exists(image_path):
                valid_anns.append(ann)
                self.image.append(image_path)

                person_id = idx  # 使用索引作为 person_id
                self.img2person.append(person_id)
                for caption in ann['captions']:
                    texts.append(caption)
                    self.txt2person.append(person_id)

                for caption_ts in ann.get('captions_bt', [''] * len(ann['captions'])):
                    texts_ts.append(caption_ts)

        if self.test_sample:
            for i in range(len(texts)):
                if random.random() < 0.0:
                    texts[i] = texts_ts[i]

            for caption in texts:
                self.text.append(pre_caption(caption, max_words))
        else:
            for caption in texts:
                self.text.append(pre_caption(caption, max_words))

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long) if self.txt2person else torch.tensor([], dtype=torch.long)
        self.img2person = torch.tensor(self.img2person, dtype=torch.long) if self.img2person else torch.tensor([], dtype=torch.long)

        if not valid_anns:
            raise ValueError("没有有效的图像路径，数据集为空")

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {e}")