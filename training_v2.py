import torch
import os
import sys
from datetime import datetime
import inspect
from datasets.steel_dataset import SeverstalDataset
from general_utils import log
from torchvision import transforms
from os.path import join, isfile
from torch.utils.data import DataLoader
from map_fusion_network import MapFusion
from models.clipseg import CLIPDensePredT
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.optim as optim
from general_utils import filter_args


def read_yaml(filepath: str) -> dict:
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def main():

    json_path = './Severstal/annotations_COCO.json'
    image_dir = './Severstal/train_subimages'
    config_path = './experiments/severstal.yaml'
    mean = [0.34388125, 0.34388125, 0.34388125]
    std = [0.13965334, 0.13965334, 0.13965334]
    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    config = read_yaml(config_path)

    n_support = config['training']['n_support']

    train_dataset = SeverstalDataset(json_path=json_path, image_dir=image_dir, image_size=(256,256),
                               n_support=n_support, transform=transform, split='train')
    val_dataset = SeverstalDataset(json_path=json_path, image_dir=image_dir, image_size=(256,256),
                               n_support=n_support, transform=transform, split='val')

    model_params = config['model_config']
    lr = config['training']['lr']
    batch_size = config['training']['batch_size']

    clipseg_model = CLIPDensePredT(**model_params).cuda()
    fusion_model = MapFusion(num_classes=4, clipseg_model=clipseg_model).cuda()
    # optimizer
    optimizer = optim.AdamW(fusion_model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

    for batch in data_loader:

        query_image = batch['query_image'].cuda()
        mask = batch['query_mask'].cuda()
        support = batch['support']
        query_class = batch['query_class'].cuda()

        optimizer.zero_grad()
        result = fusion_model.forward(query_image, support).squeeze(1).squeeze(1)
        loss = loss_fn(result, mask)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()