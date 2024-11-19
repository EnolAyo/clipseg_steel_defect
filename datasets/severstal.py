import torch
from torchvision import transforms
from datasets.utils import blend_image_segmentation

#COCO_CLASSES = {1: 'crazing', 2: 'rolled-in scale', 3: 'scratch', 4: 'inclusion', 5: 'steel plate'}
COCO_CLASSES = {1: 'network of fine, hairline cracks or fissures on the surface of the steel',
                2: 'scale—oxides embedded into the steel plate',
                3: 'shallow, narrow grooves or lines on the surface of the steel',
                4: 'impurity or foreign material embedded within the steel matrix',
                5: 'defects on a steel plate'}


class COCOWrapper(object):

    def __init__(self, split, image_size=256, aug=None, mask='text_and_blur3_highlight01', negative_prob=0,
                 with_class_label=True):
        super().__init__()

        self.mask = mask
        self.with_class_label = with_class_label
        self.negative_prob = negative_prob
        self.split = split

        from datasets.severstal_coco import DatasetCOCO
        datapath = '/home/eas/Enol/pycharm_projects/clipseg_steel_defect/Severstal/train_subimages'

        mean = [0.34388125, 0.34388125, 0.34388125]
        std = [0.13965334, 0.13965334, 0.13965334]
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.coco = DatasetCOCO(datapath, transform, split, True)

        self.all_classes = [self.coco.class_ids]
        self.coco.base_path = datapath

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, i):
        sample = self.coco[i]
        query_class = int(sample['query_class'])
        support_class = int(sample['support_class'])
        label_name = COCO_CLASSES[support_class]

        img_s, seg_s = sample['support_imgs'][0], sample['support_masks'][0]

        if query_class !=5 and self.negative_prob > 0 and torch.rand(1).item() < self.negative_prob:
            new_class_id = sample['query_class']
            new_sample_img_id = ''
            while new_class_id == sample['query_class'] or new_class_id == 5 or new_sample_img_id in self.coco.duplicates:
                sample2 = self.coco[torch.randint(0, len(self), (1,)).item()]
                new_class_id = sample2['support_class']
                new_sample_img_id = sample2['query_name']
            img_s = sample2['support_imgs'][0]
            seg_s = torch.zeros_like(seg_s)
            label_name = COCO_CLASSES[int(new_class_id)]

        mask = self.mask
        if mask == 'separate':
            supp = (img_s, seg_s)
        elif mask == 'text_label':
            # DEPRECATED
            supp = [int(sample['class_id'])]
        elif mask == 'text':
            supp = [label_name]
        else:
            if mask.startswith('text_and_'):
                mask = mask[9:]
                label_add = [label_name]
            else:
                label_add = []

            supp = label_add + blend_image_segmentation(img_s, seg_s, mode=mask)

        if self.with_class_label:
            label = (torch.zeros(0), sample['query_class'],)
        else:
            label = (torch.zeros(0),)

        return (sample['query_img'],) + tuple(supp), (sample['query_mask'].unsqueeze(0),) + label