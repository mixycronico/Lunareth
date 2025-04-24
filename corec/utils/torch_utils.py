import torch
import torch.nn as nn
from typing import Dict, Any, List
from torchvision.models import mobilenet_v3_small

def load_mobilenet_v3_small(model_path: str = None, pretrained: bool = False, n_classes: int = 3, device: str = "cpu") -> nn.Module:
    """Carga MobileNetV3 Small, preentrenado o desde un archivo."""
    try:
        model = mobilenet_v3_small(pretrained=pretrained)
        if n_classes != 1000:  # ImageNet tiene 1000 clases
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.to(device)
    except Exception as e:
        raise RuntimeError(f"Error cargando MobileNetV3 Small: {e}")

def preprocess_data(datos: Dict[str, Any], device: str) -> torch.Tensor:
    """Convierte datos de entrada a un tensor para MobileNetV3."""
    try:
        valores = datos.get("valores", [])
        if not valores:
            raise ValueError("No se proporcionaron valores en los datos")
        tensor = torch.tensor(valores, dtype=torch.float32).view(1, len(valores), 1, 1)
        return tensor.to(device)
    except Exception as e:
        raise RuntimeError(f"Error preprocesando datos: {e}")

def postprocess_logits(logits: torch.Tensor, bloque_id: str) -> List[Dict[str, Any]]:
    """Convierte logits a etiquetas y probabilidades."""
    try:
        probs = torch.softmax(logits, dim=1)
        max_probs, labels = torch.max(probs, dim=1)
        if bloque_id == "enjambre_sensor":
            etiqueta_map = {0: "normal", 1: "anomalía"}
        elif bloque_id == "crypto_trading":
            etiqueta_map = {0: "bajista", 1: "alcista", 2: "neutro"}
        elif bloque_id == "nodo_seguridad":
            etiqueta_map = {0: "riesgo_bajo", 1: "riesgo_alto"}
        else:
            etiqueta_map = {0: "desconocido"}
        return [{"etiqueta": etiqueta_map.get(label.item(), "desconocido"), "probabilidad": prob.item()}
                for prob, label in zip(max_probs, labels)]
    except Exception as e:
        raise RuntimeError(f"Error postprocesando logits: {e}")
