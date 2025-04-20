import logging
import asyncio
import time
import random
from pydantic import BaseModel, Field, ValidationError
from corec.core import ModuloBase
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad


class RedirectionConfig(BaseModel):
    """Valida configuración de redirección de entidades."""
    source_block: str
    target_block: str
    entidades: int = Field(..., ge=100, le=5000)
    canal: int = Field(..., ge=1, le=10)


class ModuloSincronizacion(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloSincronizacion")
        self.nucleus = None

    async def inicializar(self, nucleus):
        """Inicializa el módulo de sincronización."""
        self.nucleus = nucleus
        self.logger.info("[Sincronizacion] listo")

    async def fusionar_bloques(self, id1: str, id2: str, nuevo: str):
        """Fusiona dos bloques en uno nuevo."""
        reg = self.nucleus.modules["registro"].bloques
        b1, b2 = reg.get(id1), reg.get(id2)
        if b1 and b2:
            combinado = b1.entidades[:len(b1.entidades) // 2] + b2.entidades[:len(b2.entidades) // 2]
            reg[nuevo] = BloqueSimbiotico(nuevo, b1.canal, combinado, nucleus=self.nucleus)
            del reg[id1], reg[id2]
            self.logger.info(f"[Sincronizacion] {id1}+{id2}→{nuevo}")
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_fusionado",
                "nuevo_bloque": nuevo,
                "bloques_origen": [id1, id2],
                "timestamp": time.time()
            })

    async def redirigir_entidades(self, source_block: str, target_block: str, entidades: int, canal: int):
        """Redirige entidades de un bloque inactivo a un bloque activo."""
        reg = self.nucleus.modules["registro"].bloques
        source, target = reg.get(source_block), reg.get(target_block)
        if not source or not target:
            self.logger.error(
                f"[Sincronizacion] Bloques no encontrados: {source_block}, {target_block}"
            )
            return
        if source.canal != canal or target.canal != canal:
            self.logger.error(
                f"[Sincronizacion] Canales no compatibles: {source.canal} vs {target.canal}"
            )
            return
        try:
            cfg = RedirectionConfig(
                source_block=source_block,
                target_block=target_block,
                entidades=entidades,
                canal=canal
            )
            if len(source.entidades) < cfg.entidades:
                self.logger.warning(
                    f"[Sincronizacion] No suficientes entidades en {source_block}"
                )
                return
            transferidas = source.entidades[:cfg.entidades]
            source.entidades = source.entidades[cfg.entidades:]
            target.entidades.extend(transferidas)
            self.logger.info(
                f"[Sincronizacion] Redirigidas {cfg.entidades} entidades de "
                f"{source_block} a {target_block}"
            )
            await self.nucleus.publicar_alerta({
                "tipo": "entidades_redirigidas",
                "source_block": cfg.source_block,
                "target_block": cfg.target_block,
                "entidades": cfg.entidades,
                "timestamp": time.time()
            })
        except ValidationError as e:
            self.logger.error(f"[Sincronizacion] Configuración de redirección inválida: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_redireccion",
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def adaptar_bloque(self, bid: str, carga: float):
        """Adapta un bloque según fitness, carga y actividad."""
        reg = self.nucleus.modules["registro"].bloques
        b = reg.get(bid)
        if not b:
            self.logger.error(f"[Sincronizacion] No existe {bid}")
            return
        # Fusionar bloques con bajo fitness
        if b.fitness < 0.2:
            cand = next((x for x in reg.values() if x.canal == b.canal and x.fitness > 0.5), None)
            if cand:
                await self.fusionar_bloques(bid, cand.id, f"fus_{bid}_{time.time_ns()}")
        # Redirigir entidades de bloques inactivos
        if b.fitness < 0.3 and carga < 0.2 and len(b.entidades) >= 100:
            target = next((x for x in reg.values() if x.canal == b.canal and x.fitness > 0.8 and x != b), None)
            if target:
                await self.redirigir_entidades(bid, target.id, min(1000, len(b.entidades)), b.canal)
        # Dividir bloques con alta carga
        if carga > 0.8 and len(b.entidades) > 500:
            nuevo = f"{bid}_split_{time.time_ns()}"
            e2 = b.entidades[500:]
            b.entidades = b.entidades[:500]
            reg[nuevo] = BloqueSimbiotico(nuevo, b.canal, e2, nucleus=self.nucleus)
            self.logger.info(f"[Sincronizacion] Split {bid}→{nuevo}")
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_dividido",
                "nuevo_bloque": nuevo,
                "bloque_origen": bid,
                "timestamp": time.time()
            })
        # Ampliar bloques con baja carga
        elif carga < 0.2 and len(b.entidades) < 1000:
            faltan = 1000 - len(b.entidades)
            for i in range(faltan):
                async def tmp(): return {"valor": random.random()}
                b.entidades.append(crear_entidad(f"m{len(b.entidades)}", b.canal, tmp))
            self.logger.info(f"[Sincronizacion] Ampliado {bid} a {len(b.entidades)} entidades")
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_ampliado",
                "bloque_id": bid,
                "entidades": len(b.entidades),
                "timestamp": time.time()
            })

    async def ejecutar(self):
        """Ejecuta sincronización y redirección de bloques."""
        while True:
            for bid in list(self.nucleus.modules["registro"].bloques):
                await self.adaptar_bloque(bid, random.random())
            await asyncio.sleep(300)  # Sincronizar cada 5 minutos

    async def detener(self):
        """Detiene el módulo de sincronización."""
        self.logger.info("[Sincronizacion] detenido")
