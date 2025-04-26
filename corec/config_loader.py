import json
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, ValidationError, Field

class AutoreparacionConfig(BaseModel):
    max_errores: float
    min_fitness: float

class BloqueConfig(BaseModel):
    id: str
    canal: int
    entidades: int
    max_size_mb: float
    entidades_por_bloque: int
    autoreparacion: AutoreparacionConfig
    ia_timeout_seconds: float = None

class PluginBloqueConfig(BaseModel):
    bloque_id: str
    canal: int
    entidades: int
    max_size_mb: float
    max_errores: float
    min_fitness: float

class PluginConfig(BaseModel):
    enabled: bool
    path: str
    bloque: PluginBloqueConfig

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
    max_connections: int = 100
    stream_max_length: int = 5000

class IAConfig(BaseModel):
    enabled: bool
    model_path: str
    max_size_mb: float
    pretrained: bool
    n_classes: int
    timeout_seconds: float
    batch_size: int

class AnalisisDatosConfig(BaseModel):
    correlation_threshold: float
    n_estimators: int
    max_samples: int

class ConfigSchema(BaseModel):
    instance_id: str
    db_config: DBConfig
    redis_config: RedisConfig
    ia_config: IAConfig
    analisis_datos_config: AnalisisDatosConfig
    bloques: list[BloqueConfig]
    plugins: Dict[str, PluginConfig] = Field(default_factory=dict)

def load_config_dict(config_path: str) -> dict:
    """Carga y valida el archivo de configuración."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_file.open("r") as f:
            config_dict = json.load(f)

        config = ConfigSchema(**config_dict)
        return config.model_dump()
    except ValidationError as e:
        raise ValueError(f"Invalid config format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")
