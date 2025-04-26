import logging
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger("TorchUtils")

def load_mobilenet_v3_small(model_path: str = "", pretrained: bool = False, n_classes: int = 3, device: torch.device = None):
    """Carga un modelo MobileNetV3 Small."""
    try:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, n_classes)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.to(device)
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {e}")
        raise

def preprocess_data(data: dict, device: torch.device):
    """Preprocesa datos para MobileNetV3."""
    try:
        if isinstance(data, (np.ndarray, torch.Tensor)) or "image" in data:
            # Manejar imágenes o tensores
            if isinstance(data, dict):
                img = data["image"]
            else:
                img = data

            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, torch.Tensor):
                return img.to(device)  # Ya es un tensor, solo mover a dispositivo

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(img).unsqueeze(0).to(device)
        else:
            # Manejar valores numéricos
            values = np.array(data["valores"], dtype=np.float32)
            # Crear un tensor de imagen dummy (1, 3, 224, 224)
            tensor = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)
            # Rellenar el canal 0 con los valores escalados
            scale = values / (np.max(np.abs(values)) + 1e-8)  # Escalar para evitar valores extremos
            tensor[:, 0, :len(values), 0] = torch.from_numpy(scale).to(device)
            return tensor
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def postprocess_logits(logits: torch.Tensor, block_id: str) -> list:
    """Postprocesa logits para generar resultados."""
    try:
        probs = torch.softmax(logits, dim=1)
        max_probs, labels = torch.max(probs, dim=1)
        results = []
        for i, (prob, label) in enumerate(zip(max_probs, labels)):
            results.append({
                "etiqueta": f"clase_{label.item()}",
                "probabilidad": prob.item(),
                "entidad_id": f"{block_id}_ia_{i}"
            })
        return results
    except Exception as e:
        logger.error(f"Error postprocessing logits: {e}")
        raise
