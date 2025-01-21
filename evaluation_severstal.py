from models.clipseg import CLIPDensePredT
from datasets.severstal import COCOWrapper
from tqdm import tqdm
import random
import numpy as np

def evaluate(model, dataset, text_weights = 0.5):
    random.seed(33)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    score = [[], [], [], [], []]
    model.eval()
    j = 1
    with torch.no_grad():
        for data_x, data_y in tqdm(data_loader):
            j+=1
            data_x = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_x]
            data_y = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_y]

            #prompts = model.sample_prompts(data_x[1], prompt_list=('a photo of a {}',))

            prompts = data_x[1]
            visual_s_cond, _, _ = model.visual_forward(data_x[2])
            text_cond = model.compute_conditional(prompts)
            labels = data_y[2]
            cond = text_cond * text_weights + visual_s_cond * (1 - text_weights)

            preds, visual_q, _, _  = model(data_x[0], cond, return_features=True)


            new_score = get_score(preds, data_y[0], labels)
            for i, class_score in enumerate(score):
                if new_score[i] is not None:
                    score[i] += new_score[i]
    acc = []
    for i, class_score in enumerate(score):
        non_zero = sum(1 for x in class_score if x > 0)
        acc.append(non_zero / len(class_score))
        score[i] = sum(class_score) / len(class_score)
        print(len(class_score))
    return score, acc

import torch

def get_score(pred, target, query_class, threshold=0.5, eps=1e-6):

    pred = torch.sigmoid(pred)
    target = target.float()
    pred_binary = (pred >= threshold).float()
    score_per_class = []
    for class_id in range(1,6):
        class_indices = (query_class == class_id).nonzero(as_tuple=True)[0]
        if len(class_indices) == 0:
            score_per_class.append(None)
            continue

        pred_class = pred_binary[class_indices]
        mask_class = target[class_indices]


    # Compute the intersection and union for each item in the batch

        intersection = torch.sum(pred_class * mask_class, dim=(1, 2, 3))  # Sum over height, width, channels
        union = torch.sum(pred_class, dim=(1, 2, 3)) + torch.sum(mask_class, dim=(1, 2, 3))

        # Calculate Dice coefficient for each item in the batch
        score = (2 * intersection + eps) / (union + eps)
        score = score.cpu().tolist()
        score_per_class.append(score)

    # Return the mean Dice coefficient over the batch
    return score_per_class  # Convert to Python float for easier use



def main():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    weights = '/home/eas/Enol/pycharm_projects/clipseg_steel_defect/logs/rd64-7K-vit16-cbh-coco-notebooks-5classes_no_neg/weights.pth'
    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')), strict=False);
    dataset = COCOWrapper(split='test')
    model.cuda()
    text_weights = np.linspace(1, 0, 11)
    for text_weight in text_weights:
        score, acc = evaluate(model, dataset, text_weights=text_weight)
        print(f'Results for text weight: {text_weight}')
        print(f'DICE coeff.: {score}')
        print(f'Ratio of found defects: {acc}')

    """
    for i in range(1,5):
        print(f'Dice Coefficient for class {i}: {sum(results[i])/len(results[i])}')

    print(f'For images without defect, the percentage of images with defect predicted is '
          f'{sum(results[i][0])/len(results[i][0])}')
          """

if __name__ == '__main__':
    main()