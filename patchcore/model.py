import torch
import torch.nn.functional as F

def extract_features(backbone, x, layers):
    outputs = {}

    def hook(module, input, output):
        outputs[module] = output

    handles = []
    for name, module in backbone.named_modules():
        if name in layers:
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        _ = backbone(x)

    for h in handles:
        h.remove()

    feats = [outputs[m] for m in outputs]
    feats = [F.interpolate(f, size=feats[0].shape[-2:], mode="bilinear") for f in feats]
    feat = torch.cat(feats, dim=1)
    return feat