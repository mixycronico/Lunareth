import os
import json
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, root_validator
from typing import Dict, Any, Optional, List


class AutoreparacionConfig(BaseModel):
    max_errores: float = Field(gt=0, le=1.0)
    min_fitness: float = Field(ge=0, le=1.0)


class MutacionConfig(BaseModel):
    enabled: bool
    min_fitness: float = Field(ge=0, le=1.0)
    mutation_rate: float = Field(gt=0, lt=1.0)
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
    retry_delay_max: float = Field(ge=0)


class CognitivoConfig(BaseModel):
    max_memoria: int = Field(ge=1, default=1000)
    umbral_confianza: float = Field(gt=0, le=1.0, default=0.5)
    penalizacion_intuicion: float = Field(gt=0, le=1.0, default=0.9)
    max_percepciones: int = Field(ge=1, default=5000)
    impacto_adaptacion: float = Field(ge=0, le=1.0, default=0.1)
    confiabilidad_minima: float = Field(ge=0, le=1.0, default=0.4)
    umbral_afectivo_positivo: float = Field(ge=0, le=1.0, default=0.8)
    umbral_afectivo_negativo: float = Field(lt=0, ge=-1.0, default=-0.8)
    peso_afectivo: float = Field(ge=0, le=1.0, default=0.2)
    umbral_fallo: float = Field(gt=0, le=1.0, default=0.3)
    peso_semantico: float = Field(ge=0, le=1.0, default=0.1)
    umbral_cambio_significativo: float = Field(ge=0, le=1.0, default=0.05)
    tasa_aprendizaje_minima: float = Field(gt=0, le=1.0, default=0.1)
    umbral_relevancia: float = Field(ge=0, le=1.0, default=0.3)
    peso_novedad: float = Field(ge=0, le=1.0, default=0.3)


class CoreCConfig(BaseModel):
    instance_id: str
    db_config: DBConfig
    redis_config: RedisConfig
    ia_config: IAConfig
    analisis_datos_config: AnalisisDatosConfig
    ml_config: MLConfig
    autosanacion_config: AutosanacionConfig
    cognitivo_config: CognitivoConfig = Field(default_factory=CognitivoConfig)
    bloques: List[BloqueConfig]
    plugins: Dict[str, PluginConfig] = Field(default_factory=dict)
    # Parámetros por defecto relocados aquí:
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

    @root_validator(pre=True)
    def override_secrets_from_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Obliga a usar variables de entorno en producción
        db_pass = os.getenv("DB_PASSWORD")
        if db_pass:
            values["db_config"]["password"] = db_pass
        else:
            raise ValueError("Environment variable DB_PASSWORD is required")

        redis_pass = os.getenv("REDIS_PASSWORD")
        if redis_pass:
            values["redis_config"]["password"] = redis_pass
        else:
            raise ValueError("Environment variable REDIS_PASSWORD is required")

        # Validación de bloques duplicados
        bloques = values.get("bloques", [])
        ids = [b["id"] for b in bloques]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate block IDs found in configuration")

        return values


def load_config(config_path: str) -> CoreCConfig:
    """Carga y valida el archivo de configuración usando Pydantic."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        return CoreCConfig(**raw)
    except ValidationError as exc:
        # Re-lanzar con mensaje claro
        raise ValueError(f"Invalid config format:\n{exc}")
