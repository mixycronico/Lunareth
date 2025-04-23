import logging
import torch
from typing import Dict, Any
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from corec.utils.torch_utils import load_netaug_mobilev3, preprocess_data, postprocess_logits

class ModuloIA(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloIA")
        self.nucleus = None
        self.model = None
        self.device = torch.device("cpu")  # CPU para 4 núcleos

    async def inicializar(self, nucleus, config: Dict[str, Any] = None):
        """Inicializa el módulo de IA con NetAugMobileNetV3."""
        try:
            self.nucleus = nucleus
            config = config or {}
            model_path = config.get("model_path", "corec/models/mobilev3/model.pth")
            max_size_mb = config.get("max_size_mb", 0.3)
            
            # Cargar modelo
            self.model = load_netaug_mobilev3(model_path, mode="min", device=self.device)
            self.model.eval()
            self.logger.info(f"[IA] Modelo NetAugMobileNetV3 cargado desde {model_path}, modo 'min', max_size_mb: {max_size_mb}")

            # Validar recursos
            if max_size_mb < 0.3:
                self.logger.warning(f"[IA] max_size_mb ({max_size_mb}) puede ser insuficiente para NetAugMobileNetV3")
        except Exception as e:
            self.logger.error(f"[IA] Error inicializando: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_ia",
                "mensaje": str(e),
                "timestamp": random.random()
            })
            raise

    async def procesar_bloque(self, bloque: BloqueSimbiotico, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de un bloque con NetAugMobileNetV3."""
        try:
            # Preprocesar datos
            input_tensor = preprocess_data(datos, self.device)
            
            # Ejecutar inferencia
            with torch.no_grad():
                logits = self.model(input_tensor)
            
            # Postprocesar resultados
            resultados = postprocess_logits(logits, bloque.id)
            
            # Generar mensajes
            mensajes = []
            for resultado in resultados:
                mensajes.append({
                    "entidad_id": f"{bloque.id}_ia_{random.randint(0, 9999)}",
                    "canal": bloque.canal,
                    "valor": resultado.get("probabilidad", 0.0),
                    "clasificacion": resultado.get("etiqueta", ""),
                    "timestamp": time.time()
                })
            
            # Publicar alerta
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
        """Detiene el módulo de IA."""
        try:
            self.model = None
            torch.cuda.empty_cache()  # Aunque usamos CPU, por si acaso
            self.logger.info("[IA] Módulo detenido")
        except Exception as e:
            self.logger.error(f"[IA] Error deteniendo: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_detencion_ia",
                "mensaje": str(e),
                "timestamp": time.time()
            })
