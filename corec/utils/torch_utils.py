import logging
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms


logger = logging.getLogger("TorchUtils")


def load_mobilenet_v3_small(
    model_path: str = "",
    pretrained: bool = False,
    n_classes: int = 3,
    device: torch.device = None
) -> torch.nn.Module:
    """Carga un modelo MobileNetV3 Small.

    Args:
        model_path (str): Ruta al archivo de pesos del modelo.
        pretrained (bool): Si True, carga pesos preentrenados (no implementado).
        n_classes (int): Número de clases para la capa final.
        device (torch.device): Dispositivo donde cargar el modelo (CPU/GPU).

    Returns:
        torch.nn.Module: Modelo MobileNetV3 Small cargado.

    Raises:
        FileNotFoundError: Si el archivo de pesos no existe.
        RuntimeError: Si falla la carga del modelo.
    """
    try:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, n_classes)
        if model_path:
            if not torch.cuda.is_available() and device.type == "cuda":
                logger.warning("CUDA no disponible, usando CPU")
                device = torch.device("cpu")
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.to(device)
    except FileNotFoundError as e:
        logger.error(f"No se encontró el archivo de modelo en {model_path}: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Error cargando modelo desde {model_path}: {e}")
        raise


def preprocess_data(data: dict, device: torch.device) -> torch.Tensor:
    """Preprocesa datos para MobileNetV3.

    Args:
        data (dict): Diccionario con datos de entrada (valores o imagen).
        device (torch.device): Dispositivo donde procesar los datos.

    Returns:
        torch.Tensor: Tensor preprocesado listo para el modelo.

    Raises:
        ValueError: Si los datos de entrada son inválidos.
    """
    try:
        if isinstance(data, (np.ndarray, torch.Tensor)) or "image" in data:
            if isinstance(data, dict):
                img = data["image"]
            else:
                img = data

            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, torch.Tensor):
                return img.to(device)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(img).unsqueeze(0).to(device)
        else:
            values = np.array(data["valores"], dtype=np.float32)
            tensor = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)
            scale = values / (np.max(np.abs(values)) + 1e-8)
            tensor[:, 0, :len(values), 0] = torch.from_numpy(scale).to(device)
            return tensor
    except Exception as e:
        logger.error(f"Error preprocesando datos: {e}")
        raise ValueError(f"Error preprocesando datos: {e}")


def postprocess_logits(logits: torch.Tensor, block_id: str) -> list:
    """Postprocesa logits para generar resultados.

    Args:
        logits (torch.Tensor): Salida del modelo.
        block_id (str): ID del bloque asociado.

    Returns:
        list: Lista de resultados con etiquetas, probabilidades y IDs.

    Raises:
        RuntimeError: Si falla el postprocesamiento.
    """
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
        logger.error(f"Error postprocesando logits: {e}")
        raise RuntimeError(f"Error postprocesando logits: {e}")
