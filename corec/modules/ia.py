import logging
import random
import time
import asyncio
import psutil
import torch

from typing import Dict, Any, List
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from corec.utils.torch_utils import (
    load_mobilenet_v3_small,
    preprocess_data,
    postprocess_logits
)


class ModuloIA(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloIA")
        self.nucleus = None
        self.model = None
        self.device = torch.device("cpu")
        self.timeout = 2.0

    async def inicializar(self, nucleus, config: Dict[str, Any] = None):
        """Inicializa el módulo IA con MobileNetV3 y parámetros de tiempo."""
        try:
            self.nucleus = nucleus
            cfg = config or {}
            model_path = cfg.get("model_path", "")
            pretrained = cfg.get("pretrained", False)
            n_classes = cfg.get("n_classes", 3)
            self.timeout = cfg.get("timeout_seconds", 2.0)

            self.model = load_mobilenet_v3_small(
                model_path=model_path,
                pretrained=pretrained,
                n_classes=n_classes,
                device=self.device
            )
            self.logger.info(f"[IA] Modelo cargado ({'pretrained' if pretrained else model_path}), "
                             f"timeout={self.timeout}s")
        except Exception as e:
            self.logger.error(f"[IA] Error inicializando: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_ia",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def procesar_batch(
        self,
        bloque: BloqueSimbiotico,
        datos_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Realiza una única llamada al modelo para un lote de entradas,
        mide memoria y latencia, y aplica timeout + fallback.
        """
        # Monitor antes
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # MB
        t0 = time.monotonic()

        # Pre-procesado en batch
        tensors = [preprocess_data(datos, self.device) for datos in datos_list]
        batch_input = torch.cat(tensors, dim=0)

        try:
            # Inference en thread pool con timeout (circuit-breaker)
            loop = asyncio.get_event_loop()
            logits = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.model(batch_input)),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            # Fallback: mensajes vacíos o por defecto
            await self.nucleus.publicar_alerta({
                "tipo": "timeout_ia",
                "bloque_id": bloque.id,
                "timeout": self.timeout,
                "timestamp": time.time()
            })
            return [{
                "entidad_id": f"{bloque.id}_ia_fallback_{i}",
                "canal": bloque.canal,
                "valor": 0.0,
                "clasificacion": "fallback",
                "probabilidad": 0.0,
                "timestamp": time.time()
            } for i in range(len(datos_list))]

        # Post-procesado
        resultados = postprocess_logits(logits, bloque.id)

        t1 = time.monotonic()
        mem_after = process.memory_info().rss / 1024**2

        # Publicar métricas
        await self.nucleus.publicar_alerta({
            "tipo": "ia_metrics",
            "bloque_id": bloque.id,
            "latencia_ms": (t1 - t0) * 1000,
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "timestamp": time.time()
        })

        # Construir mensajes
        mensajes: List[Dict[str, Any]] = []
        for idx, res in enumerate(resultados):
            mensajes.append({
                "entidad_id": f"{bloque.id}_ia_{random.randint(0,9999)}",
                "canal": bloque.canal,
                "valor": res["probabilidad"],
                "clasificacion": res["etiqueta"],
                "probabilidad": res["probabilidad"],
                "timestamp": time.time()
            })
        return mensajes

    async def detener(self):
        """Limpia el modelo y libera caché de GPU."""
        try:
            self.model = None
            torch.cuda.empty_cache()
            self.logger.info("[IA] Módulo detenido")
        except Exception as e:
            self.logger.error(f"[IA] Error deteniendo IA: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_detencion_ia",
                "mensaje": str(e),
                "timestamp": time.time()
            })
