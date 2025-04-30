import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# Modelos Pydantic (sin cambios, se incluyen solo los relevantes para referencia)
class DBConfig(BaseModel):
    dbname: str
    user: str
    password: str
    host: str
    port: int = Field(ge=1)

class RedisConfig(BaseModel):
    host: str
    port: int = Field(ge=1)
    username: str
    password: str
    max_connections: int = Field(ge=1, default=100)
    stream_max_length: int = Field(ge=1, default=5000)

class CoreCConfig(BaseModel):
    instance_id: str
    db_config: DBConfig
    redis_config: RedisConfig
    # ... otros campos (sin cambios)
    quantization_step_default: float = Field(default=0.05, gt=0, le=1.0)
    max_enlaces_por_entidad: int = Field(default=100, ge=1)
    redis_stream_key: str = Field(default="corec:entrelazador")
    alert_threshold: float = Field(default=0.9, gt=0, le=1.0)
    max_fallos_criticos: float = Field(default=0.5, gt=0, le=1.0)
    cpu_autoadjust_threshold: float = Field(default=0.9, gt=0, le=1.0)
    ram_autoadjust_threshold: float = Field(default=0.9, gt=0, le=1.0)
    concurrent_tasks_min: int = Field(default=10, ge=1)
    concurrent_tasks_max: int = Field(default=1000, ge=1)
    concurrent_tasks_reduction_factor: float = Field(default=0.8, gt=0, le=1.0)
    concurrent_tasks_increment_factor_default: float = Field(default=1.05, gt=1.0)
    cpu_stable_cycles: int = Field(default=3, ge=1)
    cpu_readings: int = Field(default=3, ge=1)
    cpu_reading_interval: float = Field(default=0.05, gt=0)
    performance_history_size: int = Field(default=10, ge=1)
    performance_threshold: float = Field(default=0.5, gt=0)
    increment_factor_min: float = Field(default=1.01, gt=1.0)
    increment_factor_max: float = Field(default=1.1, gt=1.0)

def load_config(config_path: str) -> CoreCConfig:
    """Carga y valida el archivo de configuración."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_file.open("r") as f:
            config_dict = json.load(f)

        # Reemplazar contraseñas con variables de entorno
        config_dict["db_config"]["password"] = os.getenv("DB_PASSWORD", config_dict["db_config"]["password"])
        config_dict["redis_config"]["password"] = os.getenv("REDIS_PASSWORD", config_dict["redis_config"]["password"])

        block_ids = [block["id"] for block in config_dict.get("bloques", [])]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("Duplicate block IDs found in configuration")

        return CoreCConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid config format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")
