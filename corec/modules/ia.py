import logging
import torch
from typing import Dict, Any
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from corec.utils.torch_utils import load_mobilenet_v3_small, preprocess_data, postprocess_logits
import random
import time

class ModuloIA(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloIA")
        self.nucleus = None
        self.model = None
        self.device = torch.device("cpu")

    async def inicializar(self, nucleus, config: Dict[str, Any] = None):
        """Inicializa el módulo de IA con MobileNetV3 Small."""
        try:
            self.nucleus = nucleus
            config = config or {}
            model_path = config.get("model_path", "corec/models/mobilev3/model.pth")
            max_size_mb = config.get("max_size_mb", 50)
            pretrained = config.get("pretrained", False)
            n_classes = config.get("n_classes", 3)

            self.model = load_mobilenet_v3_small(model_path, pretrained=pretrained, n_classes=n_classes, device=self.device)
            self.logger.info(f"[IA] Modelo MobileNetV3 Small cargado desde {'ImageNet' if pretrained else model_path}, max_size_mb: {max_size_mb}")

            if max_size_mb < 10:
                self.logger.warning(f"[IA] max_size_mb ({max_size_mb}) puede ser insuficiente para MobileNetV3 Small")
        except Exception as e:
            self.logger.error(f"[IA] Error inicializando: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_ia",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def procesar_bloque(self, bloque: BloqueSimbiotico, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de un bloque con MobileNetV3 Small."""
        try:
            input_tensor = preprocess_data(datos, self.device)
            with torch.no_grad():
                logits = self.model(input_tensor)
            resultados = postprocess_logits(logits, bloque.id if bloque else "unknown")
            mensajes = []
            for resultado in resultados:
                mensajes.append({
                    "entidad_id": f"{bloque.id if bloque else 'ia'}_ia_{random.randint(0, 9999)}",
                    "canal": bloque.canal if bloque else 3,
                    "valor": resultado.get("probabilidad", 0.0),
                    "clasificacion": resultado.get("etiqueta", ""),
                    "probabilidad": resultado.get("probabilidad", 0.0),
                    "timestamp": time.time()
                })
            await self.nucleus.publicar_alerta({
                "tipo": "ia_procesado",
                "bloque_id": bloque.id if bloque else "unknown",
                "num_mensajes": len(mensajes),
                "timestamp": time.time()
            })
            self.logger.info(f"[IA] Bloque {bloque.id if bloque else 'unknown'} procesado, {len(mensajes)} mensajes generados")
            return {"mensajes": mensajes}
        except Exception as e:
            self.logger.error(f"[IA] Error procesando bloque {bloque.id if bloque else 'unknown'}: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_procesamiento_ia",
                "bloque_id": bloque.id if bloque else "unknown",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            return {"mensajes": []}

    async def detener(self):
        """Detiene el módulo de IA."""
        try:
            self.model = None
            torch.cuda.empty_cache()
            self.logger.info("[IA] Módulo detenido")
        except Exception as e:
            self.logger.error(f"[IA] Error deteniendo: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_detencion_ia",
                "mensaje": str(e),
                "timestamp": time.time()
            })
