# corec/modules/sincronizacion.py
import logging, asyncio, time, random
from corec.core import ModuloBase
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad

class ModuloSincronizacion(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloSincronizacion")
        self.nucleus = None

    async def inicializar(self, nucleus):
        self.nucleus = nucleus
        self.logger.info("[Sincronizacion] listo")

    async def fusionar_bloques(self, id1:str, id2:str, nuevo:str):
        reg = self.nucleus.modules["registro"].bloques
        b1, b2 = reg.get(id1), reg.get(id2)
        if b1 and b2:
            combinado = b1.entidades[:len(b1.entidades)//2] + b2.entidades[:len(b2.entidades)//2]
            reg[nuevo] = BloqueSimbiotico(nuevo, b1.canal, combinado, nucleus=self.nucleus)
            del reg[id1], reg[id2]
            self.logger.info(f"[Sincronizacion] {id1}+{id2}→{nuevo}")

    async def adaptar_bloque(self, bid:str, carga:float):
        reg = self.nucleus.modules["registro"].bloques
        b = reg.get(bid)
        if not b:
            self.logger.error(f"[Sincronizacion] no existe {bid}")
            return
        if b.fitness<0.2:
            cand = next((x for x in reg.values() if x.canal==b.canal and x.fitness>0.5), None)
            if cand:
                await self.fusionar_bloques(bid, cand.id, f"fus_{bid}_{time.time_ns()}")
        if carga>0.8 and len(b.entidades)>500:
            nuevo = f"{bid}_split_{time.time_ns()}"
            e2    = b.entidades[500:]
            b.entidades = b.entidades[:500]
            reg[nuevo] = BloqueSimbiotico(nuevo, b.canal, e2, nucleus=self.nucleus)
            self.logger.info(f"[Sincronizacion] split {bid}→{nuevo}")
        elif carga<0.2 and len(b.entidades)<1000:
            faltan = 1000 - len(b.entidades)
            for i in range(faltan):
                async def tmp(): return {"valor":random.random()}
                b.entidades.append(crear_entidad(f"m{len(b.entidades)}",b.canal,tmp))
            self.logger.info(f"[Sincronizacion] ampliado {bid} a {len(b.entidades)} ent.")

    async def ejecutar(self):
        while True:
            for bid in list(self.nucleus.modules["registro"].bloques):
                await self.adaptar_bloque(bid, random.random())
            await asyncio.sleep(3600)

    async def detener(self):
        self.logger.info("[Sincronizacion] detenido")