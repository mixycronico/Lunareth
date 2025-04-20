import logging
import asyncio
import random
import time
from pydantic import ValidationError
from corec.core import ModuloBase
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad
from corec.nucleus import PluginBlockConfig


class ModuloRegistro(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloRegistro")
        self.bloques = {}
        self.nucleus = None

    async def inicializar(self, nucleus):
        """Inicializa el módulo de registro y registra bloques."""
        self.nucleus = nucleus
        self.logger.info("[Registro] listo")
        for cfg in self.nucleus.config.get("bloques", []):
            await self.registrar_bloque(cfg["id"], cfg["canal"], cfg["entidades"])

    async def registrar_bloque(self, bloque_id: str, canal: int, cantidad: int):
        """Registra un bloque simbiótico con entidades especificadas."""
        try:
            # Validar configuración (reutilizamos PluginBlockConfig de nucleus)
            cfg = PluginBlockConfig(
                bloque_id=bloque_id,
                canal=canal,
                entidades=cantidad,
                max_size_mb=1,  # Valor por defecto
                max_errores=0.05,  # Valor por defecto
                min_fitness=0.2  # Valor por defecto
            )
            size = 1000
            resto = cfg.entidades
            idx = 0
            while resto > 0:
                cnt = min(size, resto)
                entidades = []
                for i in range(cnt):
                    async def tmp(): return {"valor": random.random()}
                    entidades.append(crear_entidad(f"m{idx}", cfg.canal, tmp))
                    idx += 1
                bid = bloque_id if idx == cnt else f"{bloque_id}_{idx // size}"
                bloque = BloqueSimbiotico(bid, cfg.canal, entidades, nucleus=self.nucleus)
                self.bloques[bid] = bloque
                resto -= cnt
                self.logger.info(f"[Registro] {bid} ({cnt} entidades)")
                await self.nucleus.publicar_alerta({
                    "tipo": "bloque_registrado",
                    "bloque_id": bid,
                    "entidades": cnt,
                    "canal": cfg.canal,
                    "timestamp": time.time()
                })
        except ValidationError as e:
            self.logger.error(f"[Registro] Configuración inválida para '{bloque_id}': {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_registro",
                "bloque_id": bloque_id,
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def ejecutar(self):
        """Ejecuta el registro y escritura de bloques."""
        while True:
            for b in self.bloques.values():
                await b.escribir_postgresql(self.nucleus.db_config)
            await asyncio.sleep(300)  # Escribir cada 5 minutos

    async def detener(self):
        """Detiene el módulo de registro."""
        self.logger.info("[Registro] detenido")
