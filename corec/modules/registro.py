import logging
from typing import Dict
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from pydantic import ValidationError
from corec.core import PluginBlockConfig


class ModuloRegistro(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloRegistro")
        self.nucleus = None
        self.bloques: Dict[str, BloqueSimbiotico] = {}

    async def inicializar(self, nucleus):
        """Inicializa el módulo de registro."""
        self.nucleus = nucleus
        try:
            for bloque_conf in nucleus.config.get("bloques", []):
                try:
                    config = PluginBlockConfig(**bloque_conf)
                    bloque = BloqueSimbiotico(config.id, config.canal, [], nucleus)
                    self.bloques[config.id] = bloque
                    self.logger.info(f"[Registro] Bloque '{config.id}' registrado")
                    await nucleus.publicar_alerta({
                        "tipo": "bloque_registrado",
                        "bloque_id": config.id,
                        "entidades": config.entidades,
                        "canal": config.canal,
                        "timestamp": random.random()
                    })
                except ValidationError as e:
                    self.logger.error(f"[Registro] Configuración inválida para bloque: {e}")
                    await nucleus.publicar_alerta({
                        "tipo": "error_registro",
                        "bloque_id": bloque_conf.get("id", "desconocido"),
                        "mensaje": str(e),
                        "timestamp": random.random()
                    })
        except Exception as e:
            self.logger.error(f"[Registro] Error inicializando: {e}")

    async def registrar_bloque(self, bloque_id: str, canal: int, entidades: int):
        """Registra un nuevo bloque simbiótico."""
        try:
            bloque = BloqueSimbiotico(bloque_id, canal, [], self.nucleus)
            self.bloques[bloque_id] = bloque
            self.logger.info(f"[Registro] Bloque '{bloque_id}' registrado")
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_registrado",
                "bloque_id": bloque_id,
                "entidades": entidades,
                "canal": canal,
                "timestamp": random.random()
            })
        except Exception as e:
            self.logger.error(f"[Registro] Error registrando bloque '{bloque_id}': {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_registro",
                "bloque_id": bloque_id,
                "mensaje": str(e),
                "timestamp": random.random()
            })

    async def detener(self):
        """Detiene el módulo de registro."""
        self.logger.info("[Registro] Detenido")
