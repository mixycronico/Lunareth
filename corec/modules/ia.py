import time
import asyncio
import torch
import psutil
from corec.utils.torch_utils import load_mobilenet_v3_small
from corec.blocks import BloqueSimbiotico


class ModuloIA:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enabled = False
        self.config = None
        self.nucleus = None

    async def inicializar(self, nucleus, config):
        """Inicializa el módulo de inteligencia artificial.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del módulo IA.
        """
        self.nucleus = nucleus
        self.logger = nucleus.logger
        self.config = config
        self.enabled = self.nucleus.config.ia_config.enabled
        if not self.enabled:
            self.logger.info("Módulo IA deshabilitado")
            return
        try:
            model_path = self.nucleus.config.ia_config.model_path
            self.model = load_mobilenet_v3_small(
                model_path,
                pretrained=self.nucleus.config.ia_config.pretrained,
                n_classes=self.nucleus.config.ia_config.n_classes,
                device=self.device
            )
            self.logger.info("Modelo IA inicializado")
        except Exception as e:
            self.logger.error(f"Error inicializando Módulo IA: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_ia",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def procesar_bloque(self, bloque: BloqueSimbiotico, datos: dict):
        """Procesa un bloque con el modelo de inteligencia artificial.

        Args:
            bloque (BloqueSimbiotico): Bloque a procesar (opcional).
            datos (dict): Datos de entrada para el modelo.

        Returns:
            Dict[str, Any]: Resultado con mensajes procesados.
        """
        if not self.enabled or not self.model:
            self.logger.warning("Módulo IA no habilitado o modelo no inicializado")
            return {"mensajes": []}

        mensajes = []
        try:
            cpu_percent = psutil.cpu_percent()
            mem_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
            max_size_mb = self.nucleus.config.ia_config.max_size_mb
            if cpu_percent > 90 or mem_usage_mb > max_size_mb:
                self.logger.warning(f"Recursos excedidos: CPU={cpu_percent}%, Mem={mem_usage_mb}MB")
                await self.nucleus.publicar_alerta({
                    "tipo": "alerta_recursos",
                    "mensaje": f"Recursos excedidos: CPU={cpu_percent}%, Mem={mem_usage_mb}MB",
                    "timestamp": time.time()
                })
                return {
                    "mensajes": [{
                        "clasificacion": "fallback_recursos",
                        "probabilidad": 0.0,
                        "timestamp": time.time()
                    }]
                }

            valores = datos.get("valores", [])
            if not valores:
                self.logger.debug("No hay valores para procesar")
                return {"mensajes": []}

            input_tensor = torch.tensor(valores, dtype=torch.float32).to(self.device)
            timeout_seconds = (
                bloque.ia_timeout_seconds if bloque else
                self.nucleus.config.ia_config.timeout_seconds
            )

            try:
                with torch.no_grad():
                    output = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, lambda: self.model(input_tensor)),
                        timeout=timeout_seconds
                    )
                    probs = torch.softmax(output, dim=-1)
                    clase_idx = torch.argmax(probs).item()
                    probabilidad = probs[clase_idx].item()
                    clasificacion = f"clase_{clase_idx}"
            except asyncio.TimeoutError:
                self.logger.warning("Timeout procesando bloque")
                await self.nucleus.publicar_alerta({
                    "tipo": "timeout_ia",
                    "mensaje": "Timeout procesando bloque",
                    "timestamp": time.time()
                })
                return {
                    "mensajes": [{
                        "clasificacion": "fallback",
                        "probabilidad": 0.0,
                        "timestamp": time.time()
                    }]
                }
            except RuntimeError as e:
                self.logger.error(f"Error de Torch procesando bloque: {e}")
                await self.nucleus.publicar_alerta({
                    "tipo": "error_torch",
                    "mensaje": str(e),
                    "timestamp": time.time()
                })
                return {
                    "mensajes": [{
                        "clasificacion": "error",
                        "probabilidad": 0.0,
                        "timestamp": time.time()
                    }]
                }

            mensajes.append({
                "clasificacion": clasificacion,
                "probabilidad": probabilidad,
                "timestamp": time.time()
            })
            await self.nucleus.publicar_alerta({
                "tipo": "ia_procesado",
                "mensaje": f"Procesado exitoso: {clasificacion}",
                "timestamp": time.time()
            })

        except Exception as e:
            self.logger.error(f"Error procesando bloque: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_procesamiento_ia",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            mensajes.append({
                "clasificacion": "error",
                "probabilidad": 0.0,
                "timestamp": time.time()
            })

        return {"mensajes": mensajes}

    async def detener(self):
        """Detiene el módulo de inteligencia artificial."""
        self.logger.info("Módulo IA detenido")
        self.model = None
