import torch.nn as nn
import torch
defects_description = {
        1: 'network of fine, hairline cracks or fissures on the surface of the steel',
        2: 'scale—oxides embedded into the steel plate',
        3: 'shallow, narrow grooves or lines on the surface of the steel',
        4: 'impurity or foreign material embedded within the steel matrix'
    }
class MapFusion(nn.Module):
    def __init__(self, num_classes, clipseg_model):
        super().__init__()
        self.clipseg_model = clipseg_model

        self.fusion_net = nn.Sequential(
            nn.Conv3d(1, 4,kernel_size=(4, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(4, 1, kernel_size=1)
        )

    def forward(self, query_image, support_sets):  # support_sets: dict {class_id: [(img, mask), ...]}
        prob_maps = []
        for class_id, support in support_sets.items():
            # Combine support images and masks if needed (e.g., average over shots)
            pred_prob = []
            text_prompt = defects_description[class_id]
            text_weights = torch.rand(1).cuda()
            text_cond = self.clipseg_model.compute_conditional(text_prompt)
            for visual_prompt in support:
                visual_s_cond, _, _ = self.clipseg_model.visual_forward(visual_prompt.cuda())
                cond_vector = text_cond * text_weights + visual_s_cond * (1 - text_weights)
                pred_logit, _, _, _ = self.clipseg_model(query_image.cuda(), cond_vector, return_features=True)
                pred_prob.append(torch.sigmoid(pred_logit))
            avg_prob = torch.stack(pred_prob).mean(dim=0)
            prob_maps.append(avg_prob)

        probs_stacked = torch.stack(prob_maps, dim=1)  # [B, K, H, W]
        probs_stacked = probs_stacked.permute(0, 2, 1, 3, 4)
        fused_probs = self.fusion_net(probs_stacked)   # [B, K, H, W]
        return fused_probs
