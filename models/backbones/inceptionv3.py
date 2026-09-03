import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3Baseline(nn.Module):
    """
    Inception-v3 baseline for multi-scale spatial drowsiness feature benchmark.
    Optimized with native 299x299 resolution support and optional auxiliary logits.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, aux_logits: bool = False):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        self.aux_logits = aux_logits
        init_aux = True if weights else aux_logits
        self.model = inception_v3(weights=weights, aux_logits=init_aux, transform_input=False)
        self.model.aux_logits = aux_logits
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if aux_logits and hasattr(self.model, 'AuxLogits'):
            aux_in = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(aux_in, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            if (h, w) != (299, 299):
                x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
            if self.training and self.aux_logits:
                outputs, aux_outputs = self.model(x)
                logits = outputs.view(b, t, -1).mean(dim=1)
                aux_logits = aux_outputs.view(b, t, -1).mean(dim=1)
                return {"logits": logits, "aux_logits": aux_logits, "fatigue_score": torch.softmax(logits, dim=-1)[:, -1:]}
            else:
                out = self.model(x)
                logits = out.logits if hasattr(out, 'logits') else out
                logits = logits.view(b, t, -1).mean(dim=1)
                return {"logits": logits, "fatigue_score": torch.softmax(logits, dim=-1)[:, -1:]}
        else:
            if self.training and self.aux_logits:
                outputs, aux_outputs = self.model(x)
                return {"logits": outputs, "aux_logits": aux_outputs, "fatigue_score": torch.softmax(outputs, dim=-1)[:, -1:]}
            else:
                out = self.model(x)
                logits = out.logits if hasattr(out, 'logits') else out
                return {"logits": logits, "fatigue_score": torch.softmax(logits, dim=-1)[:, -1:]}

