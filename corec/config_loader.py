# corec/config_loader.py
import json
from pathlib import Path
from pydantic import BaseModel, Field, root_validator, ValidationError
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
    max_connections: int = Field(ge=1, description="Máximo número de conexiones")
    stream_max_length: int = Field(ge=100, description="Máximo longitud del flujo de Redis")

class IAConfig(BaseModel):
    enabled: bool
    model_path: str
    max_size_mb: float = Field(ge=0.0, description="Tamaño máximo del módulo IA en MB")
    pretrained: bool
    n_classes: int = Field(ge=1, description="Número de clases para el modelo")
    timeout_seconds: float = Field(ge=0.0, description="Tiempo de espera para IA en segundos")
    batch_size: int = Field(ge=1, description="Tamaño del lote para procesamiento IA")

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
    ia_timeout_seconds: Optional[float] = Field(ge=0.0, description="Tiempo de espera para IA en segundos", default=None)

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

    @root_validator(pre=True)
    def check_path_exists(cls, values):
        path = values.get("path")
        if path and not Path(path).is_file():
            raise ValueError(f"Ruta de configuración del plugin no existe: {path}")
        return values

class CoreCConfig(BaseModel):
    instance_id: str
    db_config: DBConfig
    redis_config: RedisConfig
    ia_config: Optional[IAConfig] = None
    bloques: List[BloqueConfig]
    plugins: Dict[str, PluginConfig]

    @root_validator
    def check_unique_block_ids(cls, values):
        bloques = values.get("bloques", [])
        plugins = values.get("plugins", {})
        block_ids = [b.id for b in bloques]
        plugin_block_ids = [p.bloque.bloque_id for p in plugins.values() if p.bloque]
        all_ids = block_ids + plugin_block_ids
        duplicates = set([x for x in all_ids if all_ids.count(x) > 1])
        if duplicates:
            raise ValueError(f"IDs de bloques duplicados encontrados: {duplicates}")
        return values

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
