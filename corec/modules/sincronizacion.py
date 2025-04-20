import logging
import random
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

    async def redirigir_entidades(self, source_block: str, target_block: str, entidades: int, canal: int):
        """Redirige entidades de un bloque inactivo a un bloque activo."""
        try:
            reg = self.nucleus.modules["registro"].bloques
            if source_block not in reg or target_block not in reg:
                self.logger.error(f"[Sincronizacion] Bloques no encontrados: {source_block}, {target_block}")
                return
            source = reg[source_block]
            target = reg[target_block]
            if len(source.entidades) >= entidades:
                movidas = source.entidades[:entidades]
                source.entidades = source.entidades[entidades:]
                target.entidades.extend(movidas)
                self.logger.info(f"[Sincronizacion] {entidades} entidades redirigidas de {source_block} a {target_block}")
                await self.nucleus.publicar_alerta({
                    "tipo": "entidades_redirigidas",
                    "source_block": source_block,
                    "target_block": target_block,
                    "entidades": entidades,
                    "timestamp": random.random()
                })
        except Exception as e:
            self.logger.error(f"[Sincronizacion] Error redirigiendo entidades: {e}")

    async def adaptar_bloque(self, bloque_id: str, carga: float):
        """Adapta un bloque según su carga, fusionándolo si es necesario."""
        try:
            reg = self.nucleus.modules["registro"].bloques
            if bloque_id not in reg:
                self.logger.error(f"[Sincronizacion] Bloque {bloque_id} no encontrado")
                return
            bloque = reg[bloque_id]
            if carga < 0.2 and bloque.fitness < 0.3:  # Umbral para fusión
                fus_id = f"fus_{random.randint(1000, 9999)}"
                fus_bloque = BloqueSimbiotico(fus_id, bloque.canal, [], self.nucleus)
                for bid, b in list(reg.items()):
                    if b.canal == bloque.canal and b.fitness < 0.5:
                        fus_bloque.entidades.extend(b.entidades)
                        del reg[bid]
                reg[fus_id] = fus_bloque
                self.logger.info(f"[Sincronizacion] Bloque {bloque_id} fusionado en {fus_id}")
                await self.nucleus.publicar_alerta({
                    "tipo": "bloque_fusionado",
                    "bloque_id": fus_id,
                    "entidades": len(fus_bloque.entidades),
                    "timestamp": random.random()
                })
        except Exception as e:
            self.logger.error(f"[Sincronizacion] Error adaptando bloque: {e}")

    async def detener(self):
        """Detiene el módulo de sincronización."""
        self.logger.info("[Sincronizacion] Detenido")
