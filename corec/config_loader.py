import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class DBConfig(BaseModel):
    dbname: str
    user: str
    password: str
    host: str
    port: int

class RedisConfig(BaseModel):
    host: str
    port: int
    username: str
    password: str

class IAConfig(BaseModel):
    enabled: bool
    model_path: str
    max_size_mb: float = Field(ge=0.0, description="Tamaño máximo del módulo IA en MB")

class AutoReparacionConfig(BaseModel):
    max_errores: float = Field(ge=0.0, le=1.0, description="Máximo porcentaje de errores permitido")
    min_fitness: float = Field(ge=0.0, le=1.0, description="Fitness mínimo requerido")

class BloqueConfig(BaseModel):
    id: str
    canal: int = Field(ge=1, description="Canal del bloque")
    entidades: int = Field(ge=1, description="Número de entidades en el bloque")
    max_size_mb: float = Field(ge=0.0, description="Tamaño máximo del bloque en MB")
    entidades_por_bloque: int = Field(ge=1, description="Número de entidades por bloque")
    autoreparacion: Optional[AutoReparacionConfig] = None
    plugin: Optional[str] = None

class PluginBlockConfig(BaseModel):
    bloque_id: str
    canal: int = Field(ge=1, description="Canal del bloque del plugin")
    entidades: int = Field(ge=1, description="Número de entidades en el bloque del plugin")
    max_size_mb: float = Field(ge=0.0, description="Tamaño máximo del bloque en MB")
    max_errores: float = Field(ge=0.0, le=1.0, description="Máximo porcentaje de errores permitido")
    min_fitness: float = Field(ge=0.0, le=1.0, description="Fitness mínimo requerido")

class PluginConfig(BaseModel):
    enabled: bool
    path: str
    bloque: PluginBlockConfig

class CoreCConfig(BaseModel):
    instance_id: str
    db_config: DBConfig
    redis_config: RedisConfig
    ia_config: Optional[IAConfig] = None
    bloques: List[BloqueConfig]
    plugins: Dict[str, PluginConfig]

def load_config(path: str) -> CoreCConfig:
    """Carga y valida el JSON de configuración de CoreC."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    data = json.loads(p.read_text())
    try:
        return CoreCConfig(**data)
    except ValidationError as ve:
        print("Error en la configuración de CoreC:")
        print(ve.json())
        raise

def load_config_dict(path: str) -> dict:
    """Carga el JSON de configuración de CoreC y lo devuelve como diccionario."""
    config_obj = load_config(path)
    return config_obj.dict()
