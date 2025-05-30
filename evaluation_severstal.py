import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from map_fusion_network import MapFusion
from datasets.steel_dataset import SeverstalDataset
from models.clipseg import CLIPDensePredT
import pandas as pd

def main():

    weights_path = './weights/severstal/fusion_model_epoch_30.pth'
    batch_size = 16
    json_path = './Severstal/annotations_COCO.json'
    image_dir = './Severstal/train_subimages'
    mean = [0.34388125, 0.34388125, 0.34388125]
    std = [0.13965334, 0.13965334, 0.13965334]
    image_size = 256
    n_support = 2
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    num_classes = 5  # Change to match your model

    test_dataset = SeverstalDataset(json_path=json_path, image_dir=image_dir, image_size=(256, 256),
                                   n_support=n_support, transform=transform, split='test')
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    clipseg_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)

    # ------------------------
    # Load Model
    # ------------------------
    model = MapFusion(num_classes=num_classes, clipseg_model=clipseg_model).cuda()
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda')), strict=False)
    model.eval()

    # ------------------------
    # Evaluation
    # ------------------------
    conf_mat_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():

        for batch in tqdm(test_data_loader, desc='Evaluating'):
            query_image = batch['query_image'].cuda()
            mask = batch['query_mask'].long().cuda()
            support = batch['support']

            result = model.forward(query_image, support)
            preds = torch.argmax(result, dim=1)  # shape: [B, H, W]

            # Flatten predictions and labels
            preds_np = preds.view(-1).cpu().numpy()
            mask_np = mask.view(-1).cpu().numpy()

            # Update confusion matrix
            conf_mat_batch = confusion_matrix(mask_np, preds_np, labels=list(range(num_classes)))
            conf_mat_total += conf_mat_batch



    # ------------------------
    # Final Results
    # ------------------------


    # Confusion matrix

    print("\nConfusion Matrix:\n", conf_mat_total)

    # Global accuracy
    global_acc = np.sum(np.diag(conf_mat_total)) / np.sum(conf_mat_total)
    print(f"\nGlobal Accuracy: {global_acc:.4f}")

    # Per-class accuracy
    per_class_acc = np.diag(conf_mat_total) / (conf_mat_total.sum(axis=1) + 1e-8)
    for idx, acc in enumerate(per_class_acc):
        print(f"Class {idx} Accuracy: {acc:.4f}")

    # Per-class DICE coefficient
    TP = np.diag(conf_mat_total)
    FP = conf_mat_total.sum(axis=0) - TP
    FN = conf_mat_total.sum(axis=1) - TP

    dice_per_class = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    # Save metrics to CSV
    output_dir = "./evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix
    conf_df = pd.DataFrame(conf_mat_total,
                           index=[f"GT_{i}" for i in range(num_classes)],
                           columns=[f"Pred_{i}" for i in range(num_classes)])
    conf_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    # Save global + per-class accuracy
    metrics = {
        "Metric": ["Global Accuracy"] + [f"Class {i} Accuracy" for i in range(num_classes)] + [f"Class {i} Dice" for i
                                                                                               in range(num_classes)],
        "Value": [global_acc] + per_class_acc.tolist() + dice_per_class.tolist()
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "accuracy_metrics.csv"), index=False, float_format="%.4f")


if __name__ == '__main__':
    main()