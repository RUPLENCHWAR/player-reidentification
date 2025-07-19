import torch
import torch.nn as nn

def attempt_load(weights, map_location=None):
    # Simplified model loader for YOLOv5
    model = torch.load(weights, map_location=map_location)
    if isinstance(model, dict):
        model = model['model']
    return model.float().fuse().eval()