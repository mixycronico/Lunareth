import logging
import time
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico


class ModuloSincronizacion(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloSincronizacion")
        self.nucleus = None

    async def inicializar(self, nucleus, config=None):
        """Inicializa el módulo de sincronización.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del módulo (opcional).
        """
        try:
            self.nucleus = nucleus
            self.logger.info("Módulo Sincronización inicializado")
        except Exception as e:
            self.logger.error(f"Error inicializando Módulo Sincronización: {e}")
            raise

    async def redirigir_entidades(self, bloque_origen: BloqueSimbiotico, bloque_destino: BloqueSimbiotico, proporcion: float, canal: int):
        """Redirige entidades entre bloques.

        Args:
            bloque_origen (BloqueSimbiotico): Bloque de origen.
            bloque_destino (BloqueSimbiotico): Bloque de destino.
            proporcion (float): Proporción de entidades a redirigir.
            canal (int): Nuevo canal para las entidades.

        Raises:
            ValueError: Si no hay entidades para redirigir.
        """
        try:
            num_entidades = int(len(bloque_origen.entidades) * proporcion)
            if num_entidades <= 0:
                raise ValueError("No hay entidades para redirigir")
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
                "timestamp": time.time()
            })
            self.logger.info(
                f"{num_entidades} entidades redirigidas de {bloque_origen.id} a {bloque_destino.id}"
            )
        except Exception as e:
            self.logger.error(f"Error redirigiendo entidades: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_redireccion",
                "bloque_origen": bloque_origen.id if bloque_origen else "unknown",
                "bloque_destino": bloque_destino.id if bloque_destino else "unknown",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def adaptar_bloque(self, bloque_origen: BloqueSimbiotico, bloque_destino: BloqueSimbiotico):
        """Adapta un bloque fusionándolo con otro.

        Args:
            bloque_origen (BloqueSimbiotico): Bloque de origen.
            bloque_destino (BloqueSimbiotico): Bloque de destino.
        """
        try:
            bloque_destino.entidades.extend(bloque_origen.entidades)
            bloque_origen.entidades = []
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_adaptado",
                "bloque_origen": bloque_origen.id,
                "bloque_destino": bloque_destino.id,
                "timestamp": time.time()
            })
            self.logger.info(f"Bloque {bloque_origen.id} adaptado a {bloque_destino.id}")
        except Exception as e:
            self.logger.error(f"Error adaptando bloque: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_adaptacion",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def detener(self):
        """Detiene el módulo de sincronización."""
        self.logger.info("Módulo Sincronización detenido")
