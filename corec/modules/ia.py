import logging
import asyncio
import numpy as np
import torch
import psutil
from corec.utils.torch_utils import load_mobilenet_v3_small
from corec.blocks import BloqueSimbiotico

class ModuloIA:
    def __init__(self):
        self.logger = logging.getLogger("ModuloIA")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enabled = False
        self.config = None
        self.nucleus = None

    async def inicializar(self, nucleus, config):
        """Inicializa el módulo IA."""
        self.nucleus = nucleus
        self.config = config
        self.enabled = config.get("enabled", False)
        if not self.enabled:
            self.logger.info("[IA] Módulo deshabilitado")
            return
        try:
            model_path = config.get("model_path")
            self.model = load_mobilenet_v3_small(model_path, self.device)
            self.logger.info("[IA] Modelo inicializado")
        except Exception as e:
            self.logger.error(f"[IA] Error inicializando: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_ia",
                "mensaje": str(e),
                "timestamp": asyncio.get_event_loop().time()
            })

    async def procesar_bloque(self, bloque: BloqueSimbiotico, datos: dict):
        """Procesa un bloque con el modelo IA."""
        if not self.enabled or not self.model:
            return {"mensajes": []}
        
        mensajes = []
        try:
            # Verificar recursos
            cpu_percent = psutil.cpu_percent()
            mem_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
            max_size_mb = self.config.get("max_size_mb", 50)
            if cpu_percent > 90 or mem_usage_mb > max_size_mb:
                self.logger.warning(f"[IA] Recursos excedidos: CPU={cpu_percent}%, Mem={mem_usage_mb}MB")
                await self.nucleus.publicar_alerta({
                    "tipo": "alerta_recursos",
                    "mensaje": f"Recursos excedidos: CPU={cpu_percent}%, Mem={mem_usage_mb}MB",
                    "timestamp": asyncio.get_event_loop().time()
                })
                return {
                    "mensajes": [{
                        "clasificacion": "fallback_recursos",
                        "probabilidad": 0.0,
                        "timestamp": asyncio.get_event_loop().time()
                    }]
                }

            # Procesar datos
            valores = datos.get("valores", [])
            if not valores:
                return {"mensajes": []}

            input_tensor = torch.tensor(valores, dtype=torch.float32).to(self.device)
            timeout_seconds = bloque.ia_timeout_seconds if bloque else self.config.get("timeout_seconds", 5.0)

            try:
                # Ejecutar con timeout
                with torch.no_grad():
                    output = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, self.model, input_tensor),
                        timeout=timeout_seconds
                    )
                    probs = torch.softmax(output, dim=-1)
                    clase_idx = torch.argmax(probs).item()
                    probabilidad = probs[clase_idx].item()
                    clasificacion = f"clase_{clase_idx}"
            except asyncio.TimeoutError:
                self.logger.warning("[IA] Timeout procesando bloque")
                await self.nucleus.publicar_alerta({
                    "tipo": "timeout_ia",
                    "mensaje": "Timeout procesando bloque",
                    "timestamp": asyncio.get_event_loop().time()
                })
                return {
                    "mensajes": [{
                        "clasificacion": "fallback",
                        "probabilidad": 0.0,
                        "timestamp": asyncio.get_event_loop().time()
                    }]
                }

            mensajes.append({
                "clasificacion": clasificacion,
                "probabilidad": probabilidad,
                "timestamp": asyncio.get_event_loop().time()
            })
            await self.nucleus.publicar_alerta({
                "tipo": "ia_procesado",
                "mensaje": f"Procesado exitoso: {clasificacion}",
                "timestamp": asyncio.get_event_loop().time()
            })

        except Exception as e:
            self.logger.error(f"[IA] Error procesando bloque: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_procesamiento_ia",
                "mensaje": str(e),
                "timestamp": asyncio.get_event_loop().time()
            })
            mensajes.append({
                "clasificacion": "error",
                "probabilidad": 0.0,
                "timestamp": asyncio.get_event_loop().time()
            })

        return {"mensajes": mensajes}

    async def detener(self):
        """Detiene el módulo IA."""
        self.model = None
        self.logger.info("[IA] Módulo detenido")
