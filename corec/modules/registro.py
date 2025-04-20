import logging
import random
from typing import Dict
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from pydantic import ValidationError
from corec.core import PluginBlockConfig
from corec.entities import crear_entidad


class ModuloRegistro(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloRegistro")
        self.nucleus = None
        self.bloques: Dict[str, BloqueSimbiotico] = {}

    async def inicializar(self, nucleus):
        """Inicializa el módulo de registro."""
        self.nucleus = nucleus
        self.logger.info("[Registro] Iniciando inicialización")
        try:
            bloques_conf = nucleus.config.get("bloques", [])
            self.logger.info(f"[Registro] Procesando {len(bloques_conf)} bloques de configuración: {bloques_conf}")
            for bloque_conf in bloques_conf:
                self.logger.debug(f"[Registro] Procesando bloque: {bloque_conf}")
                try:
                    config = PluginBlockConfig(**bloque_conf)
                    self.logger.debug(f"[Registro] Configuración validada: id={config.id}, canal={config.canal}, entidades={config.entidades}")
                    entidades = [crear_entidad(f"ent_{i}", config.canal, lambda: {"valor": random.uniform(0, 1)}) for i in range(config.entidades)]
                    self.logger.debug(f"[Registro] Creadas {len(entidades)} entidades para bloque {config.id}")
                    self.logger.debug(f"[Registro] Intentando crear BloqueSimbiotico para {config.id}")
                    bloque = BloqueSimbiotico(config.id, config.canal, entidades, self.nucleus, max_size_mb=1.0)
                    self.logger.debug(f"[Registro] BloqueSimbiotico creado para {config.id}: {bloque}")
                    self.bloques[config.id] = bloque
                    self.logger.debug(f"[Registro] Bloque asignado a self.bloques[{config.id}]: {self.bloques[config.id]}")
                    self.logger.info(f"[Registro] Bloque '{config.id}' registrado")
                    await self.nucleus.publicar_alerta({
                        "tipo": "bloque_registrado",
                        "bloque_id": config.id,
                        "entidades": config.entidades,
                        "canal": config.canal,
                        "timestamp": random.random()
                    })
                except ValidationError as e:
                    self.logger.error(f"[Registro] Configuración inválida para bloque {bloque_conf.get('id', 'desconocido')}: {e}")
                    await self.nucleus.publicar_alerta({
                        "tipo": "error_registro",
                        "bloque_id": bloque_conf.get("id", "desconocido"),
                        "mensaje": str(e),
                        "timestamp": random.random()
                    })
                except Exception as e:
                    self.logger.error(f"[Registro] Error inesperado al registrar bloque {bloque_conf.get('id', 'desconocido')}: {e}")
                    await self.nucleus.publicar_alerta({
                        "tipo": "error_registro",
                        "bloque_id": bloque_conf.get("id", "desconocido"),
                        "mensaje": str(e),
                        "timestamp": random.random()
                    })
        except Exception as e:
            self.logger.error(f"[Registro] Error inicializando módulo: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion",
                "mensaje": str(e),
                "timestamp": random.random()
            })
        self.logger.info(f"[Registro] Inicialización completa. Bloques registrados: {list(self.bloques.keys())}")

    async def registrar_bloque(self, bloque_id: str, canal: int, entidades: int):
        """Registra un nuevo bloque simbiótico."""
        try:
            entidades_list = [crear_entidad(f"ent_{i}", canal, lambda: {"valor": random.uniform(0, 1)}) for i in range(entidades)]
            bloque = BloqueSimbiotico(bloque_id, canal, entidades_list, self.nucleus, max_size_mb=1.0)
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
