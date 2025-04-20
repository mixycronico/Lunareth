# corec/modules/registro.py
import logging, asyncio, random
from typing import Dict, Any

from corec.core import ModuloBase
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad

class ModuloRegistro(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloRegistro")
        self.bloques = {}
        self.nucleus = None

    async def inicializar(self, nucleus):
        self.nucleus = nucleus
        self.logger.info("[Registro] listo")
        for cfg in self.nucleus.config.get("bloques", []):
            await self.registrar_bloque(cfg["id"], cfg["canal"], cfg["entidades"])

    async def registrar_bloque(self, bloque_id:str, canal:int, cantidad:int):
        size = 1000
        resto = cantidad
        idx = 0
        while resto>0:
            cnt = min(size, resto)
            entidades = []
            for i in range(cnt):
                async def tmp(): return {"valor": random.random()}
                entidades.append(crear_entidad(f"m{idx}", canal, tmp))
                idx += 1
            bid = bloque_id if idx==cnt else f"{bloque_id}_{idx//size}"
            bloque = BloqueSimbiotico(bid, canal, entidades, nucleus=self.nucleus)
            self.bloques[bid] = bloque
            resto -= cnt
            self.logger.info(f"[Registro] {bid} ({cnt} ent.)")

    async def ejecutar(self):
        while True:
            for b in self.bloques.values():
                await b.escribir_postgresql(self.nucleus.db_config)
            await asyncio.sleep(300)

    async def detener(self):
        self.logger.info("[Registro] detenido")