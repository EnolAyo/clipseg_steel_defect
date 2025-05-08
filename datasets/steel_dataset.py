import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from datasets.utils import blend_image_segmentation


defects_description = {
                       1: 'network of fine, hairline cracks or fissures on the surface of the steel',
                       2: 'scale—oxides embedded into the steel plate',
                       3: 'shallow, narrow grooves or lines on the surface of the steel',
                       4: 'impurity or foreign material embedded within the steel matrix'
                       }

class SeverstalDataset(Dataset):
    def __init__(self, json_path, image_dir, image_size=(256, 256), n_support=1,
                 transform=None, split='train', split_ratio=0.8, seed=42):
        self.coco = COCO(json_path)
        self.image_dir = image_dir
        self.image_size = image_size
        self.n_support = n_support
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed

        # Build mapping: class_id -> list of image_ids
        self.class_to_image_ids = {cat['id']: [] for cat in self.coco.dataset['categories']}
        for ann in self.coco.dataset['annotations']:
            self.class_to_image_ids[ann['category_id']].append(ann['image_id'])

        # Deduplicate
        for k in self.class_to_image_ids:
            self.class_to_image_ids[k] = list(set(self.class_to_image_ids[k]))

        # Perform train/val split
        self.split_class_to_image_ids = self._split_dataset()

        self.category_ids = list(self.split_class_to_image_ids.keys())

    def _split_dataset(self):
        split_map = {}
        rng = random.Random(self.seed)

        for class_id, img_ids in self.class_to_image_ids.items():
            rng.shuffle(img_ids)
            split_idx = int(len(img_ids) * self.split_ratio)
            if self.split == 'train':
                split_map[class_id] = img_ids[:split_idx]
            elif self.split == 'val':
                split_map[class_id] = img_ids[split_idx:]
            else:
                raise ValueError("split must be 'train' or 'val'")

        return split_map

    def __len__(self):
        return 10000  # Random sampling

    def __getitem__(self, idx):
        query_class = random.choice(self.category_ids)
        query_images = self.split_class_to_image_ids[query_class]
        if not query_images:
            return self.__getitem__(idx)  # Retry if no data for class in split

        query_image_id = random.choice(query_images)
        query_img, query_mask = self.load_image_and_mask(query_image_id, query_class)

        support = []

        for cat_id in self.category_ids[:-1]:
            support_ids = list(set(self.split_class_to_image_ids[cat_id]) - {query_image_id})
            if not support_ids:
                continue
            chosen_ids = random.sample(support_ids, k=min(self.n_support, len(support_ids)))

            for sid in chosen_ids:
                img, mask = self.load_image_and_mask(sid, cat_id)
                supp = [defects_description[cat_id]] + blend_image_segmentation(img, mask,
                                                                              mode='blur3_highlight01')
                support.append(torch.tensor(supp))

        return {
            'query_image': query_img,
            'query_mask': query_mask,
            'support': torch.stack(support),
            'query_class': torch.tensor(query_class, dtype=torch.long)
        }

    def load_image_and_mask(self, image_id, category_id):
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB').resize(self.image_size)

        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=category_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros(self.image_size, dtype=np.uint8)

        for ann in anns:
            m = mask_utils.decode(ann['segmentation'])
            m = Image.fromarray(m).resize(self.image_size, resample=Image.NEAREST)
            mask = np.maximum(mask, np.array(m))

        mask = torch.tensor(mask, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, mask

