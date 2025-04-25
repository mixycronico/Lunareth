# corec/utils/torch_utils.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
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

def preprocess_data(datos: dict, device: torch.device) -> torch.Tensor:
    """Preprocesa datos de entrada para MobileNetV3."""
    try:
        # Assume datos contains an 'image' key with a numpy array [H, W, 3]
        image = datos.get("image")
        if image is None:
            raise ValueError("No 'image' key found in datos")
        
        # Convert to numpy array if not already
        image = np.asarray(image, dtype=np.float32)
        
        # Ensure image is [H, W, 3]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image shape [H, W, 3], got {image.shape}")
        
        # Convert to [3, H, W] and normalize to [0, 1]
        image = image.transpose(2, 0, 1) / 255.0
        
        # Convert to tensor and apply MobileNetV3 preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [3, H, W] to [3, H, W]
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(torch.from_numpy(image)).to(device)
        return tensor.unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {e}")

def postprocess_logits(logits: torch.Tensor, bloque_id: str) -> list[dict]:
    """Postprocesa logits para obtener probabilidades y etiquetas."""
    try:
        probabilities = torch.softmax(logits, dim=1)
        max_probs, predicted = torch.max(probabilities, dim=1)
        
        resultados = []
        for i, (prob, pred) in enumerate(zip(max_probs, predicted)):
            resultados.append({
                "probabilidad": prob.item(),
                "etiqueta": f"class_{pred.item()}",
                "bloque_id": bloque_id
            })
        return resultados
    except Exception as e:
        raise RuntimeError(f"Error postprocessing logits: {e}")
