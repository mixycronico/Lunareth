import json
import os
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv


# Cargar variables de entorno desde .env
load_dotenv()


class AutoreparacionConfig(BaseModel):
    max_errores: float = Field(gt=0, le=1.0)
    min_fitness: float = Field(ge=0, le=1.0)


class MutacionConfig(BaseModel):
    enabled: bool
    min_fitness: float = Field(ge=0, le=1.0)
    mutation_rate: float = Field(gt=0, le=1.0)
    ml_enabled: Optional[bool] = False


class AutorreplicacionConfig(BaseModel):
    enabled: bool
    max_entidades: int = Field(ge=1)
    min_fitness_trigger: float = Field(ge=0, le=1.0)


class BloqueConfig(BaseModel):
    id: str
    canal: int = Field(ge=1)
    entidades: int = Field(ge=1)
    max_size_mb: float = Field(gt=0)
    entidades_por_bloque: int = Field(ge=1)
    quantization_step: float = Field(gt=0, le=1.0)
    max_concurrent_tasks: Optional[int] = Field(ge=1, default=None)
    cpu_intensive: bool = False
    ia_timeout_seconds: Optional[float] = Field(gt=0, default=None)
    autoreparacion: AutoreparacionConfig
    mutacion: MutacionConfig
    autorreplicacion: AutorreplicacionConfig


class PluginBloqueConfig(BaseModel):
    bloque_id: str
    canal: int = Field(ge=1)
    entidades: int = Field(ge=1)
    max_size_mb: float = Field(gt=0)
    max_errores: float = Field(gt=0, le=1.0)
    min_fitness: float = Field(ge=0, le=1.0)
    quantization_step: float = Field(gt=0, le=1.0)
    max_concurrent_tasks: int = Field(ge=1)
    cpu_intensive: bool
    mutacion: MutacionConfig
    autorreplicacion: AutorreplicacionConfig


class PluginConfig(BaseModel):
    enabled: bool
    path: str
    bloque: PluginBloqueConfig


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


class IAConfig(BaseModel):
    enabled: bool
    model_path: str
    max_size_mb: float = Field(gt=0)
    pretrained: bool
    n_classes: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    batch_size: int = Field(ge=1)


class AnalisisDatosConfig(BaseModel):
    correlation_threshold: float = Field(gt=0, le=1.0)
    n_estimators: int = Field(ge=1)
    max_samples: int = Field(ge=1)


class MLConfig(BaseModel):
    enabled: bool
    model_type: str
    historial_size: int = Field(ge=1)
    min_samples_train: int = Field(ge=1)


class AutosanacionConfig(BaseModel):
    enabled: bool
    check_interval_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=1)
    retry_delay_min: float = Field(gt=0)
    retry_delay_max: float = Field(gt=0)


class CognitivoConfig(BaseModel):
    max_memoria: int = Field(ge=1, default=1000)
    umbral_confianza: float = Field(gt=0, le=1.0, default=0.5)
    penalizacion_intuicion: float = Field(gt=0, le=1.0, default=0.9)
    max_percepciones: int = Field(ge=1, default=5000)
    impacto_adaptacion: float = Field(ge=0, le=1.0, default=0.1)
    confiabilidad_minima: float = Field(gt=0, le=1.0, default=0.4)
    umbral_afectivo_positivo: float = Field(gt=0, le=1.0, default=0.8)
    umbral_afectivo_negativo: float = Field(lt=0, default=-0.8)
    peso_afectivo: float = Field(ge=0, le=1.0, default=0.2)
    umbral_fallo: float = Field(gt=0, le=1.0, default=0.3)
    peso_semantico: float = Field(ge=0, le=1.0, default=0.1)
    umbral_cambio_significativo: float = Field(ge=0, le=1.0, default=0.05)
    tasa_aprendizaje_minima: float = Field(gt=0, le=1.0, default=0.1)
    umbral_relevancia: float = Field(gt=0, le=1.0, default=0.3)
    peso_novedad: float = Field(ge=0, le=1.0, default=0.3)


class CoreCConfig(BaseModel):
    instance_id: str
    db_config: DBConfig
    redis_config: RedisConfig
    ia_config: IAConfig
    analisis_datos_config: AnalisisDatosConfig
    ml_config: MLConfig
    autosanacion_config: AutosanacionConfig
    cognitivo_config: CognitivoConfig = Field(default_factory=lambda: CognitivoConfig())
    bloques: list[BloqueConfig]
    plugins: Dict[str, PluginConfig] = Field(default_factory=dict)
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
