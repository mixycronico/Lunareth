# corec/utils/torch_utils.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def load_mobilenet_v3_small(
    model_path: str = None,
    pretrained: bool = False,
    n_classes: int = 3,
    device: str = "cpu"
) -> nn.Module:
    """Carga MobileNetV3 Small con cuantización dinámica."""
    try:
        model = mobilenet_v3_small(pretrained=pretrained)
        if n_classes != 1000:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        if model_path:
            try:
                state = torch.load(model_path, map_location=device)
                model.load_state_dict(state)
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        model.eval()
        return model.to(device)
    except Exception as e:
        raise RuntimeError(f"Error cargando MobileNetV3 Small: {e}")
