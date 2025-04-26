import logging
import random
import time
import asyncio
import psutil
import torch
from typing import Dict, Any, List
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from corec.utils.torch_utils import load_mobilenet_v3_small, preprocess_data, postprocess_logits

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
            self.timeout = cfg.get("timeout_seconds", 2.0)
            if not cfg.get("enabled", False):
                self.logger.info("[IA] Módulo IA deshabilitado")
                return

            model_path = cfg.get("model_path", "")
            pretrained = cfg.get("pretrained", False)
            n_classes = cfg.get("n_classes", 3)

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

    async def procesar_batch(self, bloque: BloqueSimbiotico, datos_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Realiza una única llamada al modelo para un lote de entradas."""
        process = psutil.Process()
        cpu_percent = psutil.cpu_percent()
        mem_before = process.memory_info().rss / 1024**2
        t0 = time.monotonic()

        max_cpu = 90.0
        max_mem = 1000
        if cpu_percent > max_cpu or mem_before > max_mem:
            self.logger.warning(f"[IA] Recursos excedidos: CPU={cpu_percent}%, Mem={mem_before}MB")
            await self.nucleus.publicar_alerta({
                "tipo": "alerta_recursos",
                "bloque_id": bloque.id,
                "cpu_percent": cpu_percent,
                "mem_mb": mem_before,
                "mensaje": "Uso de recursos excedido",
                "timestamp": time.time()
            })
            return [{
                "entidad_id": f"{bloque.id}_ia_fallback_{i}",
                "canal": bloque.canal,
                "valor": 0.0,
                "clasificacion": "fallback_recursos",
                "probabilidad": 0.0,
                "timestamp": time.time()
            } for i in range(len(datos_list))]

        timeout = bloque.ia_timeout_seconds if hasattr(bloque, 'ia_timeout_seconds') and bloque.ia_timeout_seconds else self.timeout
        try:
            tensors = [preprocess_data(datos, self.device) for datos in datos_list]
            batch_input = torch.cat(tensors, dim=0)
        except Exception as e:
            self.logger.error(f"[IA] Error preprocessing data: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_preprocesamiento_ia",
                "bloque_id": bloque.id,
                "mensaje": str(e),
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

        for attempt in range(3):
            try:
                loop = asyncio.get_event_loop()
                logits = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self.model(batch_input)),
                    timeout=timeout
                )
                break
            except asyncio.TimeoutError:
                self.logger.warning(f"[IA] Timeout en intento {attempt + 1}/{3}, timeout={timeout}s")
                await self.nucleus.publicar_alerta({
                    "tipo": "timeout_ia",
                    "bloque_id": bloque.id,
                    "timeout": timeout,
                    "attempt": attempt + 1,
                    "timestamp": time.time()
                })
                if attempt == 2:
                    return [{
                        "entidad_id": f"{bloque.id}_ia_fallback_{i}",
                        "canal": bloque.canal,
                        "valor": 0.0,
                        "clasificacion": "fallback",
                        "probabilidad": 0.0,
                        "timestamp": time.time()
                    } for i in range(len(datos_list))]
        else:
            return []

        try:
            resultados = postprocess_logits(logits, bloque.id)
        except Exception as e:
            self.logger.error(f"[IA] Error postprocessing logits: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_postprocesamiento_ia",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
            return []

        t1 = time.monotonic()
        mem_after = process.memory_info().rss / 1024**2

        await self.nucleus.publicar_alerta({
            "tipo": "ia_metrics",
            "bloque_id": bloque.id,
            "latencia_ms": (t1 - t0) * 1000,
            "cpu_percent": cpu_percent,
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "timestamp": time.time()
        })

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

    async def procesar_bloque(self, bloque: BloqueSimbiotico, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa un bloque con MobileNetV3 Small."""
        try:
            mensajes = await self.procesar_batch(bloque, [datos])
            await self.nucleus.publicar_alerta({
                "tipo": "ia_procesado",
                "bloque_id": bloque.id,
                "num_mensajes": len(mensajes),
                "timestamp": time.time()
            })
            self.logger.info(f"[IA] Bloque {bloque.id} procesado, {len(mensajes)} mensajes generados")
            return {"mensajes": mensajes}
        except Exception as e:
            self.logger.error(f"[IA] Error procesando bloque {bloque.id}: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_procesamiento_ia",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
            return {"mensajes": []}

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
