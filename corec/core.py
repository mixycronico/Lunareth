import yaml
from pydantic import BaseModel, Field


class ComponenteBase:
    async def inicializar(self, nucleus):
        pass

    async def manejar_comando(self, comando):
        pass

    async def detener(self):
        pass


class PluginBlockConfig(BaseModel):
    id: str
    canal: int = Field(ge=1)
    entidades: int = Field(ge=1)


class PluginCommand(BaseModel):
    action: str
    params: dict = {}


def cargar_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            return config
    except Exception as e:
        print(f"Error cargando configuración: {e}")
        return {}
