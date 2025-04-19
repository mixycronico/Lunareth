#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corec/bootstrap.py
Orquestador plug-and-play para CoreC.
"""

from corec.core import asyncio, logging, json, Path, importlib, cargar_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

class Bootstrap:
    def __init__(self, config_path: str = "configs/corec_config.json", instance_id: str = "corec1"):
        self.logger = logging.getLogger("Bootstrap")
        self.instance_id = instance_id
        self.config_path = config_path
        self.components = {}
        self.config = cargar_config(config_path)

    def load_component(self, directorio: str, nombre: str):
        directorio_path = Path(__file__).parent / directorio
        if not directorio_path.exists():
            self.logger.warning(f"Directorio {directorio} no existe")
            return
        try:
            modulo = importlib.import_module(f"corec.{directorio}.{nombre}")
            instancia = modulo.Componente(self.config_path, self.instance_id)
            self.components[nombre] = instancia
            self.logger.info(f"Componente {nombre} cargado desde {directorio}")
        except Exception as e:
            self.logger.error(f"Error cargando componente {nombre}: {e}")

    async def inicializar(self):
        self.load_component("", "nucleus")
        for archivo in (Path(__file__).parent / "modules").glob("*.py"):
            if not archivo.name.startswith("__"):
                self.load_component("modules", archivo.stem)
        for archivo in (Path(__file__).parent / "plugins").glob("*.py"):
            if not archivo.name.startswith("__"):
                self.load_component("plugins", archivo.stem)
        for componente in self.components.values():
            await componente.inicializar()

    async def iniciar(self):
        await self.inicializar()
        tasks = [componente.ejecutar() for componente in self.components.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def detener(self):
        for componente in self.components.values():
            await componente.detener()
        self.logger.info("Sistema CoreC detenido")

async def main():
    bootstrap = Bootstrap(config_path="configs/corec_config.json", instance_id="corec1")
    try:
        await bootstrap.iniciar()
    except KeyboardInterrupt:
        await bootstrap.detener()

if __name__ == "__main__":
    asyncio.run(main())