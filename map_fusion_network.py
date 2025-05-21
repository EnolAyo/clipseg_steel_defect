import torch.nn as nn
import torch
defects_description = {
        1: 'network of fine, hairline cracks or fissures on the surface of the steel',
        2: 'scale—oxides embedded into the steel plate',
        3: 'shallow, narrow grooves or lines on the surface of the steel',
        4: 'impurity or foreign material embedded within the steel matrix'
    }
class MapFusion(nn.Module):
    def __init__(self, num_classes, clipseg_model_class_1, clipseg_model_class_2, clipseg_model_class_3,
                 clipseg_model_class_4):
        super().__init__()
        self.clipseg_model_class_1 = clipseg_model_class_1
        self.clipseg_model_class_2 = clipseg_model_class_2
        self.clipseg_model_class_3 = clipseg_model_class_3
        self.clipseg_model_class_4 = clipseg_model_class_4


        self.fusion_net = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
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
            text_cond = self.clipseg_model_class_1.compute_conditional(text_prompt)
            for visual_prompt in support:
                visual_s_cond, _, _ = self.clipseg_model_class_1.visual_forward(visual_prompt.cuda())
                cond_vector = text_cond * text_weights + visual_s_cond * (1 - text_weights)
                if class_id == 1:
                    pred_logit, _, _, _ = self.clipseg_model_class_1(query_image.cuda(), cond_vector, return_features=True)
                elif class_id == 2:
                    pred_logit, _, _, _ = self.clipseg_model_class_2(query_image.cuda(), cond_vector, return_features=True)
                elif class_id == 3:
                    pred_logit, _, _, _ = self.clipseg_model_class_3(query_image.cuda(), cond_vector, return_features=True)
                else:
                    pred_logit, _, _, _ = self.clipseg_model_class_4(query_image.cuda(), cond_vector, return_features=True)

                output_maps.append(pred_logit)
            avg_logits_per_class = torch.stack(output_maps).mean(dim=0)
            logit_maps.append(avg_logits_per_class)
        output = torch.cat(logit_maps, dim=1)
        output = self.fusion_net(output)

        return output
