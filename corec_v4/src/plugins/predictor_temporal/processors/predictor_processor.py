#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/predictor_temporal/processors/predictor_processor.py
"""
predictor_processor.py
Genera predicciones de series temporales usando LSTM, ajustadas por datos macroeconómicos, incluyendo DXY dinámico.
"""

from ....core.processors.base import ProcesadorBase
from ....core.entidad_base import Event
from ....utils.logging import logger
from ..utils.db import PredictorDB
import torch
import torch.nn as nn
import redis.asyncio as aioredis
import psycopg2
import json
import zstandard as zstd
from typing import Dict, Any
from datetime import datetime, timedelta
import backoff
import numpy as np

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class PredictorProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("PredictorProcessor")
        self.lstm_window = config.get("predictor_config", {}).get("lstm_window", 60)
        self.lstm_hidden_size = config.get("predictor_config", {}).get("lstm_hidden_size", 50)
        self.lstm_layers = config.get("predictor_config", {}).get("lstm_layers", 2)
        self.max_datos = config.get("predictor_config", {}).get("max_datos", 1000)
        self.model_path = config.get("predictor_config", {}).get("model_path", "models/lstm_model.pth")
        self.retrain_interval = config.get("predictor_config", {}).get("retrain_interval", 86400)
        self.circuit_breaker = config.get("config", {}).get("circuit_breaker", {})
        self.plugin_db = None
        self.model = None
        self.price_history = {}
        self.macro_context = {}
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.predictions_count = 0
        self.last_retrain_time = datetime.utcnow()
        self.failure_count = 0
        self.breaker_tripped = False
        self.breaker_reset_time = None

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.plugin_db = PredictorDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a predictor_db, usando almacenamiento temporal")
            await self.nucleus.publicar_alerta({"tipo": "db_connection_error", "plugin": "predictor_temporal", "message": "No se pudo conectar a predictor_db"})

        redis_config = self.config.get("redis", {})
        self.redis = await aioredis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            decode_responses=True
        )

        self.model = LSTMPredictor(input_size=1, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layers)
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            self.logger.info("Modelo LSTM cargado desde %s", self.model_path)
        except FileNotFoundError:
            self.logger.warning("Modelo no encontrado, usando pesos iniciales")
        self.model.eval()

        try:
            monitor_db_config = {
                "host": "monitor_db",
                "port": 5432,
                "database": "monitor_db",
                "user": "monitor_user",
                "password": "secure_password"
            }
            with psycopg2.connect(**monitor_db_config) as conn:
                cur = conn.cursor()
                symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
                for symbol in symbols:
                    cur.execute(
                        "SELECT price FROM market_data WHERE symbol = %s ORDER BY timestamp DESC LIMIT %s",
                        (symbol, self.lstm_window)
                    )
                    prices = [row[0] for row in cur.fetchall()]
                    if prices:
                        self.price_history[symbol] = prices[::-1]
                        self.logger.info(f"Cargados {len(prices)} precios históricos para {symbol}")
                    else:
                        self.logger.warning(f"No se encontraron datos históricos para {symbol}")
                        await self.nucleus.publicar_alerta({"tipo": "no_historical_data", "plugin": "predictor_temporal", "message": f"No se encontraron datos históricos para {symbol}"})
                cur.close()
        except Exception as e:
            self.logger.error(f"Error cargando datos históricos: {e}")
            await self.nucleus.publicar_alerta({"tipo": "historical_data_error", "plugin": "predictor_temporal", "message": str(e)})

        self.logger.info("PredictorProcessor inicializado")

    async def check_circuit_breaker(self) -> bool:
        if self.breaker_tripped:
            now = datetime.utcnow()
            if now >= self.breaker_reset_time:
                self.breaker_tripped = False
                self.failure_count = 0
                self.breaker_reset_time = None
                self.logger.info("Circuit breaker reseteado")
            else:
                self.logger.warning("Circuit breaker activo hasta %s", self.breaker_reset_time)
                return False
        return True

    async def register_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.circuit_breaker.get("max_failures", 3):
            self.breaker_tripped = True
            self.breaker_reset_time = datetime.utcnow() + timedelta(seconds=self.circuit_breaker.get("reset_timeout", 900))
            self.logger.error("Circuit breaker activado hasta %s", self.breaker_reset_time)
            await self.nucleus.publicar_alerta({"tipo": "circuit_breaker_tripped", "plugin": "predictor_temporal"})

    async def procesar(self, datos: Any, contexto: Dict[str, Any]) -> Any:
        if not await self.check_circuit_breaker():
            return {"estado": "error", "mensaje": "Circuit breaker activo"}

        symbol = datos.get("symbol", "UNKNOWN")
        valores = self.price_history.get(symbol, datos.get("valores", []))[:self.max_datos]
        if not valores:
            return {"estado": "error", "mensaje": "No hay datos para procesar"}

        try:
            input_data = torch.tensor(valores[-self.lstm_window:], dtype=torch.float32).reshape(1, self.lstm_window, 1)
            with torch.no_grad():
                prediction = self.model(input_data).detach().numpy().tolist()[0][0]

            # Ajustar con contexto macro
            macro_adjustment = 1.0
            if self.macro_context.get("sp500_price", 0) > 0:
                macro_adjustment += (self.macro_context["sp500_price"] / 4500 - 1) * 0.1
            if self.macro_context.get("vix_price", 0) > 20:
                macro_adjustment *= 0.95
            if self.macro_context.get("gold_price", 0) > 1850:
                macro_adjustment *= 0.98
            if self.macro_context.get("dxy_change_percent", 0) != 0:
                # Factor dinámico basado en correlación DXY-BTC
                correlation = self.macro_context.get("dxy_btc_correlation", -0.6)
                dxy_impact = correlation * self.macro_context["dxy_change_percent"] * 0.03
                macro_adjustment *= (1 + dxy_impact)

            adjusted_prediction = prediction * macro_adjustment

            analisis = await self.nucleus.razonar(
                {"valores": valores, "symbol": symbol, "macro_context": self.macro_context},
                f"Análisis predictivo para {symbol} en {contexto['canal']}"
            )
            resultado_analisis = analisis.get("respuesta", "Análisis no disponible")

            actual_value = datos.get("actual_value")
            error = abs(adjusted_prediction - actual_value) if actual_value is not None else None
            if error is not None:
                self.mse_sum += error ** 2
                self.mae_sum += error
                self.predictions_count += 1

            datos_comprimidos = zstd.compress(json.dumps({"symbol": symbol, "prediction": adjusted_prediction, "mse": error ** 2 if error else None}).encode())
            await self.redis_client.xadd(f"corec_stream_{contexto['instance_id']}", {"data": datos_comprimidos})

            if self.plugin_db and self.plugin_db.conn:
                await self.plugin_db.save_prediction(
                    nano_id=contexto["nano_id"],
                    symbol=symbol,
                    prediction={"value": adjusted_prediction},
                    actual_value=actual_value,
                    error=error,
                    macro_context=self.macro_context,
                    timestamp=contexto["timestamp"]
                )

            if self.predictions_count > 0 and self.predictions_count % 10 == 0:
                mse = self.mse_sum / self.predictions_count
                mae = self.mae_sum / self.predictions_count
                await self.plugin_db.save_metrics(
                    nano_id=contexto["nano_id"],
                    mse=mse,
                    mae=mae,
                    predictions_count=self.predictions_count,
                    timestamp=contexto["timestamp"]
                )
                self.mse_sum = 0.0
                self.mae_sum = 0.0
                self.predictions_count = 0

            if (datetime.utcnow() - self.last_retrain_time).total_seconds() > self.retrain_interval:
                await self.retrain_model(valores)

            return {
                "estado": "ok",
                "symbol": symbol,
                "prediction": adjusted_prediction,
                "analisis": resultado_analisis,
                "timestamp": contexto["timestamp"],
                "error": error
            }
        except Exception as e:
            self.logger.error("Error procesando datos: %s", e)
            await self.register_failure()
            return {"estado": "error", "mensaje": str(e)}

    async def retrain_model(self, valores: list) -> None:
        try:
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            data = torch.tensor(valores, dtype=torch.float32).reshape(-1, 1)
            for _ in range(10):
                optimizer.zero_grad()
                inputs = data[:-1].reshape(1, -1, 1)
                targets = data[1:].reshape(1, -1, 1)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            torch.save(self.model.state_dict(), self.model_path)
            self.last_retrain_time = datetime.utcnow()
            self.model.eval()
            self.logger.info("Modelo reentrenado y guardado")
        except Exception as e:
            self.logger.error(f"Error reentrenando modelo: {e}")
            await self.nucleus.publicar_alerta({"tipo": "retrain_error", "plugin": "predictor_temporal", "message": str(e)})

    async def manejar_evento(self, event: Event) -> None:
        try:
            datos = json.loads(zstd.decompress(event.datos["data"]))
            if event.canal == "macro_data":
                self.macro_context = datos
                self.logger.debug("Contexto macro actualizado: %s", datos)
            elif event.canal == "market_data":
                symbol = datos.get("symbol")
                price = datos.get("price")
                if symbol and price:
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    self.price_history[symbol].append(price)
                    self.price_history[symbol] = self.price_history[symbol][-self.max_datos:]
                    result = await self.procesar(datos, {"canal": event.canal, "instance_id": self.nucleus.instance_id, "nano_id": "predictor_temporal", "timestamp": datos["timestamp"]})
                    self.logger.debug("Predicción generada para %s: %s", symbol, result)
            elif event.canal == "predictor_temporal" and datos.get("action") == "retrain":
                await self.retrain_model(self.price_history.get(datos.get("symbol", "BTC/USDT"), []))
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            await self.register_failure()
            await self.nucleus.publicar_alerta({"tipo": "event_error", "plugin": "predictor_temporal", "message": str(e)})

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        await self.redis.close()
        self.logger.info("PredictorProcessor detenido")