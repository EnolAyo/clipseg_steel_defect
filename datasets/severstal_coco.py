r""" COCO-20i few-shot semantic segmentation dataset """
import os
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
from collections import Counter
import random


class DatasetCOCO(Dataset):
    def __init__(self, datapath, transform, split, use_original_imgsize):
        self.split = split
        self.nclass = 5
        self.benchmark = 'coco_severstal'
        self.base_path = datapath
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.class_ids = [1, 2, 3, 4, 5]
        self.img_metadata = self.build_img_metadata()
        self.len = self.__len__()
        self.duplicates = self.find_duplicates()
        self.ids_by_class = self.get_anns_id_by_cat()

    def __len__(self):
        return len(self.img_metadata.anns)


    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, support_class, org_qry_imsize = self.load_frame(idx)

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'query_class': torch.tensor(class_sample),
                 'support_class': torch.tensor(support_class),}

        return batch


    def build_img_metadata(self):
        split = 'val' if self.split == 'test' else self.split
        coco = COCO(f"/home/eas/Enol/pycharm_projects/clipseg_steel_defect/Severstal/annotations_COCO_{split}.json")
        return coco

    def get_anns_id_by_cat(self):
        ids_by_class = {1: [], 2: [], 3: [], 4: [], 5: []}
        for i in self.img_metadata.anns:
            ann = self.img_metadata.anns[i]
            img_class = ann['category_id']
            ann_id = ann['id']
            ids_by_class[img_class].append(ann_id)
        return ids_by_class

    def find_duplicates(self):
        id_list = []
        for i in range(len(self.img_metadata.anns)):
            id_list.append(self.img_metadata.anns[i]['image_id'])

        counts = Counter(id_list)
        duplicates = [item for item, frequency in counts.items() if frequency > 1]
        return duplicates


    def read_mask(self, rle_code):
        binary_mask = mask_util.decode(rle_code)
        binary_mask[binary_mask != 0] = 1
        mask = torch.tensor(binary_mask)
        return mask

    def load_frame(self, idx):
        metadata = self.img_metadata
        support_names = []
        support_imgs = []
        support_masks = []
        if self.split in ['train', 'val']:
            query_class = random.randint(1, 5)
            ann_id = random.choice(self.ids_by_class[query_class])
            query = metadata.loadAnns(ann_id)[0]
            query_name = query['image_id']
            query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
            rle_mask = query['segmentation']
            query_mask = self.read_mask(rle_mask)
            org_qry_imsize = query_img.size
            support_class = query_class if query_class in [1, 2, 3, 4] else random.randint(1, 4)
            while True:
                support_id = random.choice(self.ids_by_class[support_class])
                if ann_id != support_id:
                    break
            support = metadata.loadAnns(support_id)[0]

        else:  # for testing we iterate over all annotations
            query = metadata.anns[idx]
            query_name = query['image_id']
            query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
            query_class = query['category_id']
            query_id = query['id']
            org_qry_imsize = query_img.size
            query_mask = self.read_mask(query['segmentation'])
            support_class = query_class if query_class in [1, 2, 3, 4] else random.randint(1, 4)
            support_id = query_id
            random.seed(33)
            while query_id == support_id:
                support_id = random.choice(self.ids_by_class[support_class])
            support = metadata.loadAnns(ids=support_id)[0]


        support_name = support['image_id']
        support_img = Image.open(os.path.join(self.base_path, support_name)).convert('RGB')
        support_mask_rle = support['segmentation']
        support_mask = self.read_mask(support_mask_rle)
        support_masks.append(support_mask)
        support_names.append(support_name)
        support_imgs.append(support_img)


        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, query_class, support_class, org_qry_imsize

