import torch.nn as nn
import torch
defects_description = {
        0: 'steel',
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
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )



    def forward(self, query_image, support_sets):  # support_sets: dict {class_id: [(img, mask), ...]}
        logit_maps = []
        for class_id, support in support_sets.items():
            # Combine support images and masks if needed (e.g., average over shots)
            output_maps = []
            text_prompt = defects_description[class_id]
            text_weights = torch.rand(1).cuda()
            text_weights = 1
            text_cond = self.clipseg_model.compute_conditional(text_prompt)

            for visual_prompt in support:
                visual_s_cond, _, _ = self.clipseg_model.visual_forward(visual_prompt.cuda())
                cond_vector = text_cond * text_weights + visual_s_cond * (1 - text_weights)
                pred_logit, _, _, _ = self.clipseg_model(query_image.cuda(), cond_vector, return_features=True)
                output_maps.append(pred_logit)

            avg_logits_per_class = torch.stack(output_maps).mean(dim=0)
            logit_maps.append(avg_logits_per_class)
        output = torch.cat(logit_maps, dim=1)
        output = self.fusion_net(output)

        return output
