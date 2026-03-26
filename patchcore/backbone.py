import torch
import torchvision.models as models

def get_backbone(name: str, pretrained=True):
    if name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    model.eval()
    return model