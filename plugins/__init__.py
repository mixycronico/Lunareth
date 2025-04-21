from pydantic import BaseModel, Field


class PluginBlockConfig(BaseModel):
    id: str = Field(..., description="Identificador único del bloque")
    canal: int = Field(..., description="Canal del bloque")
    entidades: int = Field(..., description="Número de entidades en el bloque")
    max_size_mb: float = Field(default=10.0, description="Tamaño máximo del bloque en MB")


class PluginCommand(BaseModel):
    action: str = Field(..., description="Acción a ejecutar por el plugin")
    params: dict = Field(default_factory=dict, description="Parámetros adicionales para la acción")
