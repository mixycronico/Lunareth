#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/predictor_processor.py
Genera predicciones de series temporales usando LSTM, ajustadas por datos macroeconómicos.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import torch
import torch.nn as nn
import json
import numpy as np
import asyncio
import backoff
from typing import Dict, Any, List
from datetime import datetime, timedelta

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

class PredictorProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("PredictorProcessor")
        self.lstm_window = self.config.get("predictor_config", {}).get("lstm_window", 60)
        self.lstm_hidden_size = self.config.get("predictor_config", {}).get("lstm_hidden_size", 50)
        self.lstm_layers = self.config.get("predictor_config", {}).get("lstm_layers", 2)
        self.max_datos = self.config.get("predictor_config", {}).get("max_datos", 1000)
        self.model_path = self.config.get("predictor_config", {}).get("model_path", "models/lstm_model.pth")
        self.retrain_interval = self.config.get("predictor_config", {}).get("retrain_interval", 86400)
        self.circuit_breaker = CircuitBreaker(
            self.config.get("predictor_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
            self.config.get("predictor_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
        )
        self.plugin_db = TradingDB(self.config.get("db_config", {}))
        self.model = LSTMPredictor(input_size=1, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layers)
        self.price_history = {}
        self.macro_context = {}
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.predictions_count = 0
        self.last_retrain_time = datetime.utcnow()

    async def inicializar(self):
        await self.plugin_db.connect()
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            self.logger.info(f"Modelo LSTM cargado desde {self.model_path}")
        except FileNotFoundError:
            self.logger.warning("Modelo no encontrado, usando pesos iniciales")
        self.model.eval()
        try:
            cur = self.conn.cursor()
            symbols = self.config.get("monitor_config", {}).get("symbols", [])
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
                    await self.nucleus.publicar_alerta({
                        "tipo": "no_historical_data",
                        "plugin": "crypto_trading",
                        "message": f"No se encontraron datos históricos para {symbol}"
                    })
            cur.close()
        except Exception as e:
            self.logger.error(f"Error cargando datos históricos: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "historical_data_error",
                "plugin": "crypto_trading",
                "message": str(e)
            })
        self.logger.info("PredictorProcessor inicializado")

    async def procesar(self, datos: Dict[str, Any], contexto: Dict[str, Any]) -> Dict[str, Any]:
        if not self.circuit_breaker.check():
            return {"estado": "error", "mensaje": "Circuit breaker activo"}
        symbol = datos.get("symbol", "UNKNOWN")
        valores = self.price_history.get(symbol, datos.get("valores", []))[:self.max_datos]
        if not valores or len(valores) < self.lstm_window:
            return {"estado": "error", "mensaje": f"Datos insuficientes para {symbol}"}
        try:
            input_data = torch.tensor(valores[-self.lstm_window:], dtype=torch.float32).reshape(1, self.lstm_window, 1)
            with torch.no_grad():
                prediction = self.model(input_data).detach().numpy().tolist()[0][0]
            macro_adjustment = 1.0
            if self.macro_context.get("sp500_price", 0) > 0:
                macro_adjustment += (self.macro_context["sp500_price"] / 4500 - 1) * 0.1
            if self.macro_context.get("vix_price", 0) > 20:
                macro_adjustment *= 0.95
            if self.macro_context.get("gold_price", 0) > 1850:
                macro_adjustment *= 0.98
            if self.macro_context.get("dxy_change_percent", 0) != 0:
                correlation = self.macro_context.get("dxy_btc_correlation", -0.6)
                dxy_impact = correlation * self.macro_context["dxy_change_percent"] * 0.03
                macro_adjustment *= (1 + dxy_impact)
            adjusted_prediction = prediction * macro_adjustment
            actual_value = datos.get("actual_value")
            error = abs(adjusted_prediction - actual_value) if actual_value is not None else None
            if error is not None:
                self.mse_sum += error ** 2
                self.mae_sum += error
                self.predictions_count += 1
            prediction_data = {"symbol": symbol, "prediction": adjusted_prediction, "mse": error ** 2 if error else None}
            datos_comprimidos = zstd.compress(json.dumps(prediction_data).encode())
            mensaje = await serializar_mensaje(int(contexto["timestamp"] % 1000000), self.canal, adjusted_prediction, True)
            await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
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
                "timestamp": contexto["timestamp"],
                "error": error
            }
        except Exception as e:
            self.logger.error(f"Error procesando datos: {e}")
            self.circuit_breaker.register_failure()
            return {"estado": "error", "mensaje": str(e)}

    async def retrain_model(self, valores: List[float]) -> None:
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
            self.logger.info(f"Modelo reentrenado y guardado en {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error reentrenando modelo: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "retrain_error",
                "plugin": "crypto_trading",
                "message": str(e)
            })

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "macro_data":
                self.macro_context = mensaje
                self.logger.debug(f"Contexto macro actualizado: {mensaje}")
            elif mensaje.get("tipo") == "market_data":
                symbol = mensaje.get("symbol")
                price = mensaje.get("price")
                if symbol and price:
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    self.price_history[symbol].append(price)
                    self.price_history[symbol] = self.price_history[symbol][-self.max_datos:]
                    result = await self.procesar(mensaje, {
                        "tipo": "market_data",
                        "instance_id": self.nucleus.instance_id,
                        "nano_id": "predictor_temporal",
                        "timestamp": mensaje["timestamp"]
                    })
                    self.logger.debug(f"Predicción generada para {symbol}: {result}")
            elif mensaje.get("tipo") == "predictor_temporal" and mensaje.get("action") == "retrain":
                await self.retrain_model(self.price_history.get(mensaje.get("symbol", "BTC/USDT"), []))
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            self.circuit_breaker.register_failure()

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("PredictorProcessor detenido")