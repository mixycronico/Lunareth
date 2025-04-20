import logging
import yaml
from typing import Dict, Any
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class ComponenteBase(ABC):
    """Clase base abstracta para componentes del sistema CoreC."""
    @abstractmethod
    async def inicializar(self, nucleus):
        pass

    @abstractmethod
    async def detener(self):
        pass


def cargar_config(config_path: str) -> Dict[str, Any]:
    """Carga la configuración desde un archivo YAML."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or {}
        logging.getLogger("CoreC").info(f"[Core] Configuración cargada desde {config_path}")
        return config
    except Exception as e:
        logging.getLogger("CoreC").error(f"[Core] Error cargando configuración: {e}")
        return {}


class PluginBlockConfig(BaseModel):
    """Configuración para un bloque simbiótico de un plugin."""
    id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    canal: int = Field(..., ge=1)
    entidades: int = Field(..., ge=100)
    bloque_id: str = None

    class Config:
        extra = "forbid"

    def __init__(self, **data):
        super().__init__(**data)
        if self.bloque_id is None:
            self.bloque_id = f"{self.id}_block"


class PluginCommand(BaseModel):
    """Modelo para comandos enviados a plugins."""
    action: str
    params: Dict[str, Any] = {}

    class Config:
        extra = "forbid"
