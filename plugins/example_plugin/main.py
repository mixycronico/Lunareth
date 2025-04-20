import logging
import random
from corec.core import ComponenteBase


class ExamplePlugin(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ExamplePlugin")
        self.nucleus = None
        self.config = None

    async def inicializar(self, nucleus, config=None):
        """Inicializa el plugin con el núcleo y una configuración opcional."""
        self.nucleus = nucleus
        self.config = config or {}
        self.logger.info("[ExamplePlugin] Inicializado")

    async def manejar_comando(self, comando: dict) -> dict:
        """Maneja comandos enviados al plugin."""
        try:
            action = comando.get("action")
            params = comando.get("params", {})
            if action == "procesar_bloque":
                bloque_id = params.get("bloque_id")
                result = await self.procesar_bloque(bloque_id)
                self.logger.info(f"[ExamplePlugin] Comando procesar_bloque ejecutado: {bloque_id}")
                return result
            else:
                self.logger.error(f"[ExamplePlugin] Acción no soportada: {action}")
                return {"status": "error", "message": "Acción no soportada"}
        except Exception as e:
            self.logger.error(f"[ExamplePlugin] Error manejando comando: {e}")
            return {"status": "error", "message": str(e)}

    async def procesar_bloque(self, bloque_id: str) -> dict:
        """Procesa un bloque simbiótico."""
        try:
            self.logger.info(f"[ExamplePlugin] Procesando bloque: {bloque_id}")
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_procesado",
                "bloque_id": bloque_id,
                "timestamp": random.random()
            })
            return {"status": "success", "bloque_id": bloque_id}
        except Exception as e:
            self.logger.error(f"[ExamplePlugin] Error procesando bloque: {e}")
            return {"status": "error", "message": str(e)}

    async def detener(self):
        """Detiene el plugin."""
        self.logger.info("[ExamplePlugin] Detenido")
