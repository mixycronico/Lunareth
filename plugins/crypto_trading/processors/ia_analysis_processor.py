#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/ia_analysis_processor.py

Analiza volúmenes, precios y datos macroeconómicos usando IA avanzada (CNN-LSTM-Transformer con métodos Bayesianos),
para generar recomendaciones automáticas para trading dinámico.
"""
from corec.core import ComponenteBase  # Eliminamos serializar_mensaje de aquí
from corec.messages import serializar_mensaje  # Importamos directamente desde corec.messages
from ..utils.helpers import CircuitBreaker
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from torch.distributions import Normal
import zstd

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)

class AdvancedTradingModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_heads=4):
        super(AdvancedTradingModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.transformer = TransformerEncoder(hidden_size, num_heads, num_layers)
        self.fc_mean = nn.Linear(hidden_size, 1)
        self.fc_std = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, seq_len, input_size] -> [batch, input_size, seq_len]
        x = F.relu(self.conv1d(x))  # [batch, hidden_size, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_size]
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch, seq_len, hidden_size]
        transformer_out = self.transformer(lstm_out)  # [batch, seq_len, hidden_size]
        final_out = transformer_out[:, -1, :]  # [batch, hidden_size]
        mean = torch.sigmoid(self.fc_mean(final_out))  # Probabilidad de subida (0 a 1)
        std = F.softplus(self.fc_std(final_out))  # Desviación estándar (positiva)
        return mean, std

class IAAnalisisProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.logger = logging.getLogger("IAAnalisisProcessor")
        self.circuit_breaker = CircuitBreaker(3, 900)
        self.model = AdvancedTradingModel(input_size=12)  # 12 inputs: volumen, cambio de precio, VIX, DXY, altcoins, RSI, MACD
        self.model_path = "plugins/crypto_trading/models/ia_analisis.pth"
        self.input_history = []
        self.max_data_points = 100
        self.training_data = []  # Para reentrenamiento dinámico

    async def inicializar(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            self.logger.info("Modelo IA cargado")
        except FileNotFoundError:
            self.logger.warning("Modelo IA no encontrado, inicializando nuevo")
        self.logger.info("IAAnalisisProcessor inicializado")

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: List[float]) -> float:
        if len(prices) < 26:
            return 0.0
        exp12 = np.convolve(prices, np.ones(12)/12, mode='valid')[-1]
        exp26 = np.convolve(prices, np.ones(26)/26, mode='valid')[-1]
        macd = exp12 - exp26
        return macd

    async def procesar_datos(self, inputs: List[float]) -> Dict[str, Any]:
        inputs_tensor = torch.tensor([inputs], dtype=torch.float32).unsqueeze(1)  # [batch=1, seq_len=1, input_size=12]
        self.model.eval()
        with torch.no_grad():
            mean, std = self.model(inputs_tensor)
            prediction = mean.item()  # Probabilidad de subida (0 a 1)
            confidence = 1.0 - std.item()  # Intervalo de confianza (0 a 1)
        return {"prediccion": prediction, "confianza": confidence, "accion": "comprar" if prediction > 0.5 else "vender"}

    async def run_analysis_loop(self):
        while True:
            if not self.circuit_breaker.check():
                await asyncio.sleep(60)
                continue

            exchange_data = await self.redis_client.get("exchange_data")
            macro_data = await self.redis_client.get("macro_data")
            prices_data = await self.redis_client.get("market:history:BTCUSDT")

            if exchange_data and macro_data and prices_data:
                exchange_json = json.loads(exchange_data)
                macro_json = json.loads(macro_data)
                prices = [float(x) for x in json.loads(prices_data)] if prices_data else [50000] * 50

                # Calcular indicadores técnicos
                rsi = self.calculate_rsi(prices)
                macd = self.calculate_macd(prices)

                inputs = [
                    exchange_json["volume"],
                    exchange_json["price_change"],
                    macro_json["vix"],
                    macro_json["dxy_change"],
                    rsi,
                    macd
                ]
                inputs += [exchange_json.get(f"altcoin_{i}_volume", 0) for i in range(6)]

                if len(self.input_history) >= self.max_data_points:
                    self.input_history.pop(0)
                self.input_history.append(inputs)

                # Agregar datos para reentrenamiento
                self.training_data.append(inputs)
                if len(self.training_data) > 1000:
                    self.training_data.pop(0)

                recommendation = await self.procesar_datos(inputs)
                recommendation["timestamp"] = datetime.utcnow().timestamp()

                datos_comprimidos = zstd.compress(json.dumps(recommendation).encode())
                mensaje = await serializar_mensaje(int(recommendation["timestamp"] % 1000000), self.canal, 0.0, True)

                await self.redis_client.xadd("crypto_trading_decisions", {"data": mensaje})

                self.logger.info(f"Recomendación IA generada: {recommendation}")

            await asyncio.sleep(60)

    async def retrain_loop(self):
        """Reentrena el modelo cada 24 horas con datos recientes."""
        while True:
            await asyncio.sleep(86400)  # 24 horas
            if len(self.training_data) < 100:
                self.logger.warning("Datos insuficientes para reentrenamiento")
                continue

            try:
                self.logger.info("Iniciando reentrenamiento del modelo...")
                self.model.train()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                criterion = nn.MSELoss()

                # Preparar datos de entrenamiento (simulados aquí, en producción usar etiquetas reales)
                inputs = torch.tensor(self.training_data, dtype=torch.float32).unsqueeze(1)  # [batch, seq_len=1, input_size=12]
                targets = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(len(self.training_data))], dtype=torch.float32).unsqueeze(-1)

                for epoch in range(10):
                    optimizer.zero_grad()
                    mean, _ = self.model(inputs)
                    loss = criterion(mean, targets)
                    loss.backward()
                    optimizer.step()
                    self.logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

                torch.save(self.model.state_dict(), self.model_path)
                self.logger.info("Modelo reentrenado y guardado")
            except Exception as e:
                self.logger.error(f"Error al reentrenar el modelo: {e}")

    async def detener(self):
        torch.save(self.model.state_dict(), self.model_path)
        self.logger.info("IAAnalisisProcessor detenido")
