import torch
from datasets.steel_dataset import SeverstalDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from map_fusion_network import MapFusion
from models.clipseg import CLIPDensePredT
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.optim as optim
import os


def read_yaml(filepath: str) -> dict:
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def main():
    os.makedirs('./weights/severstal', exist_ok=True)
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

    max_epochs = 100

    n_support = config['training']['n_support']

    train_dataset = SeverstalDataset(json_path=json_path, image_dir=image_dir, image_size=(256,256),
                               n_support=n_support, transform=transform, split='train')
    val_dataset = SeverstalDataset(json_path=json_path, image_dir=image_dir, image_size=(256,256),
                               n_support=n_support, transform=transform, split='val')
    val_interval = config['training']['val_interval']

    model_params = config['model_config']
    lr = config['training']['lr']
    batch_size = config['training']['batch_size']

    clipseg_model = CLIPDensePredT(**model_params).cuda()

    fusion_model = MapFusion(num_classes=5, clipseg_model=clipseg_model).cuda()
    # optimizer
    optimizer = optim.AdamW(fusion_model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    epoch = 0
    writer = SummaryWriter(log_dir='./logs/severstal')
    while epoch <= max_epochs:
        n_batch = 0
        epoch_loss = 0
        for batch in train_data_loader:
            fusion_model.train()
            query_image = batch['query_image'].cuda()
            mask = batch['query_mask'].long().cuda()
            support = batch['support']
            #query_class = batch['query_class'].cuda()

            optimizer.zero_grad()
            result = fusion_model.forward(query_image, support)

            loss = loss_fn(result, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1

        # Log and print training loss
        epoch_loss /= n_batch
        writer.add_scalar("loss/train", epoch_loss, epoch)
        print(f"[Epoch {epoch}] Train Loss: {epoch_loss:.4f}")

        # Validate
        fusion_model.eval()
        with torch.no_grad():
            val_loss_total = 0
            val_batches = 0
            if epoch % val_interval == 0:
                for val_batch in val_data_loader:
                    val_query = val_batch['query_image'].cuda()
                    val_mask = val_batch['query_mask'].long().cuda()
                    val_support = val_batch['support']
                    val_output = fusion_model(val_query, val_support)

                    val_loss = loss_fn(val_output, val_mask)
                    val_loss_total += val_loss.item()
                    val_batches += 1

                avg_val_loss = val_loss_total / val_batches
                writer.add_scalar("loss/val", avg_val_loss, epoch)
                print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")

            torch.save(fusion_model.state_dict(), f'./weights/severstal/fusion_model_epoch_{epoch}.pth')
            epoch += 1


if __name__ == '__main__':
    main()