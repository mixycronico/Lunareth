import torch
import torch.nn as nn
from typing import Dict, Any, List
from models.netaug.netaug_mbv3 import NetAugMobileNetV3
from models.base.mbv3 import MobileNetV3

def load_netaug_mobilev3(model_path: str, mode: str = "min", device: str = "cpu") -> nn.Module:
    """Carga NetAugMobileNetV3 desde un archivo .pth."""
    try:
        # Configuración para clasificación (ajusta n_classes según necesidades)
        model = NetAugMobileNetV3(
            base_net=MobileNetV3(),
            aug_expand_list=[1.0],
            aug_width_mult_list=[0.5],
            n_classes=3,  # Ejemplo: "alcista", "bajista", "neutro" para crypto_trading
            dropout_rate=0.0
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.set_active(mode=mode)
        model.eval()
        return model.to(device)
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo desde {model_path}: {e}")

def preprocess_data(datos: Dict[str, Any], device: str) -> torch.Tensor:
    """Convierte datos de entrada a un tensor para NetAugMobileNetV3."""
    try:
        valores = datos.get("valores", [])
        if not valores:
            raise ValueError("No se proporcionaron valores en los datos")
        # Convertir a tensor [batch=1, channels=len(valores), height=1, width=1]
        tensor = torch.tensor(valores, dtype=torch.float32).view(1, len(valores), 1, 1)
        return tensor.to(device)
    except Exception as e:
        raise RuntimeError(f"Error preprocesando datos: {e}")

def postprocess_logits(logits: torch.Tensor, bloque_id: str) -> List[Dict[str, Any]]:
    """Convierte logits a etiquetas y probabilidades."""
    try:
        probs = torch.softmax(logits, dim=1)
        max_probs, labels = torch.max(probs, dim=1)
        # Mapa de etiquetas según el bloque
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
