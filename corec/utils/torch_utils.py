# corec/utils/torch_utils.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import numpy as np

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

def preprocess_data(image: Image.Image) -> torch.Tensor:
    """Preprocesa una imagen para MobileNetV3."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def postprocess_logits(logits: torch.Tensor) -> np.ndarray:
    """Postprocesa logits para obtener probabilidades."""
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probabilities
