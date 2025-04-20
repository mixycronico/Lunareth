import logging
import random
from typing import Dict
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad


class ModuloSincronizacion(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloSincronizacion")
        self.nucleus = None

    async def inicializar(self, nucleus, config=None):
        """Inicializa el módulo de sincronización."""
        try:
            self.nucleus = nucleus
            self.logger.info("[Sincronización] Módulo inicializado")
        except Exception as e:
            self.logger.error(f"[Sincronización] Error inesperado al inicializar: {e}")

    async def redirigir_entidades(self, bloque_origen: BloqueSimbiotico, bloque_destino: BloqueSimbiotico, proporcion: float, canal: int):
        """Redirige entidades entre bloques."""
        try:
            num_entidades = int(len(bloque_origen.entidades) * proporcion)
            entidades_a_mover = bloque_origen.entidades[:num_entidades]
            bloque_origen.entidades = bloque_origen.entidades[num_entidades:]
            bloque_destino.entidades.extend(entidades_a_mover)
            for entidad in entidades_a_mover:
                entidad.canal = canal
            await self.nucleus.publicar_alerta({
                "tipo": "entidades_redirigidas",
                "bloque_origen": bloque_origen.id,
                "bloque_destino": bloque_destino.id,
                "num_entidades": num_entidades,
                "timestamp": random.random()
            })
            self.logger.info(f"[Sincronización] {num_entidades} entidades redirigidas de {bloque_origen.id} a {bloque_destino.id}")
        except Exception as e:
            self.logger.error(f"[Sincronización] Error redirigiendo entidades: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_redireccion",
                "mensaje": str(e),
                "timestamp": random.random()
            })
            raise

    async def adaptar_bloque(self, bloque_origen: BloqueSimbiotico, bloque_destino: BloqueSimbiotico):
        """Adapta un bloque fusionándolo con otro."""
        try:
            bloque_destino.entidades.extend(bloque_origen.entidades)
            bloque_origen.entidades = []
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_adaptado",
                "bloque_origen": bloque_origen.id,
                "bloque_destino": bloque_destino.id,
                "timestamp": random.random()
            })
            self.logger.info(f"[Sincronización] Bloque {bloque_origen.id} adaptado a {bloque_destino.id}")
        except Exception as e:
            self.logger.error(f"[Sincronización] Error adaptando bloque: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_adaptacion",
                "mensaje": str(e),
                "timestamp": random.random()
            })
            raise

    async def detener(self):
        """Detiene el módulo de sincronización."""
        self.logger.info("[Sincronización] Módulo detenido")
