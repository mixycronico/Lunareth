import logging
from typing import Dict
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico


class ModuloSincronizacion(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloSincronizacion")
        self.nucleus = None

    async def inicializar(self, nucleus):
        """Inicializa el módulo de sincronización."""
        self.nucleus = nucleus
        self.logger.info("[Sincronizacion] Inicializado")

    async def redirigir_entidades(self, origen_id: str, destino_id: str, num_entidades: int, canal: int):
        """Redirige entidades de un bloque a otro."""
        try:
            registro = self.nucleus.modules["registro"]
            origen = registro.bloques.get(origen_id)
            destino = registro.bloques.get(destino_id)
            if not origen or not destino:
                self.logger.error(f"[Sincronizacion] Bloques {origen_id} o {destino_id} no encontrados")
                return
            if len(origen.entidades) < num_entidades:
                self.logger.error(f"[Sincronizacion] No hay suficientes entidades en {origen_id}")
                return
            entidades_redirigidas = origen.entidades[:num_entidades]
            origen.entidades = origen.entidades[num_entidades:]
            destino.entidades.extend(entidades_redirigidas)
            self.logger.info(f"[Sincronizacion] Redirigidas {num_entidades} entidades de {origen_id} a {destino_id}")
            await self.nucleus.publicar_alerta({
                "tipo": "entidades_redirigidas",
                "origen_id": origen_id,
                "destino_id": destino_id,
                "num_entidades": num_entidades,
                "timestamp": random.random()
            })
        except Exception as e:
            self.logger.error(f"[Sincronizacion] Error redirigiendo entidades: {e}")

    async def adaptar_bloque(self, bloque_id: str, carga: float):
        """Adapta un bloque fusionándolo con otro si tiene bajo fitness."""
        try:
            registro = self.nucleus.modules["registro"]
            bloque = registro.bloques.get(bloque_id)
            if not bloque:
                self.logger.error(f"[Sincronizacion] Bloque {bloque_id} no encontrado")
                return
            if bloque.fitness < 0.2 and carga < 0.5:
                otros_bloques = [b for bid, b in registro.bloques.items() if bid != bloque_id]
                if not otros_bloques:
                    self.logger.info(f"[Sincronizacion] No hay otros bloques para fusionar con {bloque_id}")
                    return
                bloque_mejor = max(otros_bloques, key=lambda b: b.fitness)
                nuevo_id = f"fus_{random.randint(1000, 9999)}"
                nuevas_entidades = bloque.entidades + bloque_mejor.entidades
                nuevo_bloque = BloqueSimbiotico(nuevo_id, bloque.canal, nuevas_entidades, bloque.max_size_mb, self.nucleus)
                registro.bloques[nuevo_id] = nuevo_bloque
                del registro.bloques[bloque_id]
                del registro.bloques[bloque_mejor.id]
                self.logger.info(f"[Sincronizacion] Bloques {bloque_id} y {bloque_mejor.id} fusionados en {nuevo_id}")
                await self.nucleus.publicar_alerta({
                    "tipo": "bloques_fusionados",
                    "bloque_id": nuevo_id,
                    "origen_ids": [bloque_id, bloque_mejor.id],
                    "timestamp": random.random()
                })
            else:
                self.logger.info(f"[Sincronizacion] No se requiere fusión para {bloque_id}")
        except Exception as e:
            self.logger.error(f"[Sincronizacion] Error adaptando bloque {bloque_id}: {e}")

    async def detener(self):
        """Detiene el módulo de sincronización."""
        self.logger.info("[Sincronizacion] Detenido")
