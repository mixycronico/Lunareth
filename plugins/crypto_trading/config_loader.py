import json
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict

# --- Modelos Pydantic que reflejan tu schema JSON ---

class CircuitBreakerConfig(BaseModel):
    max_failures: int = Field(..., description="Máximo de fallos antes de disparar el breaker")
    reset_timeout: int = Field(..., description="Segundos para reiniciar el breaker")

class Exchange(BaseModel):
    name: str
    api_key: str
    api_secret: str
    symbols: List[str]

class ExchangeConfig(BaseModel):
    exchanges: List[Exchange]
    fetch_interval: int
    circuit_breaker: CircuitBreakerConfig

class PhaseConfig(BaseModel):
    name: str
    min: float
    max: float
    risk_per_trade: float

class CapitalConfig(BaseModel):
    min_contribution: float
    max_active_ratio: float
    phases: List[PhaseConfig]
    circuit_breaker: CircuitBreakerConfig

class SettlementConfig(BaseModel):
    settlement_time: str
    circuit_breaker: CircuitBreakerConfig

class MacroConfig(BaseModel):
    symbols: List[str]
    altcoin_symbols: List[str]
    update_interval: int
    api_keys: Dict[str, str]
    circuit_breaker: CircuitBreakerConfig

class MonitorConfig(BaseModel):
    symbols: List[str]
    update_interval: int
    circuit_breaker: CircuitBreakerConfig

class PredictorConfig(BaseModel):
    lstm_window: int
    lstm_hidden_size: int
    lstm_layers: int
    max_datos: int
    model_path: str
    retrain_interval: int
    circuit_breaker: CircuitBreakerConfig

class AnalyzerConfig(BaseModel):
    analysis_interval: int
    auto_execute: bool
    circuit_breaker: CircuitBreakerConfig

class ExecutionConfig(BaseModel):
    risk_per_trade: float
    take_profit: float
    stop_loss: float
    circuit_breaker: CircuitBreakerConfig

class UserConfig(BaseModel):
    jwt_secret: str
    circuit_breaker: CircuitBreakerConfig

class DBConfig(BaseModel):
    dbname: str
    user: str
    password: str
    host: str
    port: str

class CryptoTradingConfig(BaseModel):
    exchange_config: ExchangeConfig
    capital_config: CapitalConfig
    settlement_config: SettlementConfig
    macro_config: MacroConfig
    monitor_config: MonitorConfig
    predictor_config: PredictorConfig
    analyzer_config: AnalyzerConfig
    execution_config: ExecutionConfig
    user_config: UserConfig
    db_config: DBConfig

class AppConfig(BaseModel):
    crypto_trading: CryptoTradingConfig

def load_config(path: str = "config.json") -> AppConfig:
    """Carga y valida el JSON de configuración."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    data = json.loads(p.read_text())
    try:
        return AppConfig(**data)
    except ValidationError as ve:
        print("Error en la configuración:")
        print(ve.json())
        raise

def load_config_dict(path: str = "config.json") -> dict:
    """Carga el JSON de configuración y lo devuelve como diccionario."""
    config_obj = load_config(path)
    return config_obj.crypto_trading.dict()
