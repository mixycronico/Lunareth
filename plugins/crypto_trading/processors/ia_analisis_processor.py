#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/ia_analisis_processor.py

Analiza volúmenes, precios y datos macroeconómicos usando IA (LSTM/Transformer), 
para generar recomendaciones automáticas para trading dinámico.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.helpers import CircuitBreaker
import torch
import torch.nn as nn
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class IAAnalisisProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("IAAnalisisProcessor")
        self.circuit_breaker = CircuitBreaker(3, 900)
        self.model = SimpleLSTM(input_size=10)
        self.model_path = "plugins/crypto_trading/models/ia_analisis.pth"
        self.input_history = []
        self.max_data_points = 100

    async def inicializar(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            self.logger.info("Modelo IA cargado")
        except FileNotFoundError:
            self.logger.warning("Modelo IA no encontrado, inicializando nuevo")
        asyncio.create_task(self.run_analysis_loop())
        self.logger.info("IAAnalisisProcessor inicializado")

    async def procesar_datos(self, inputs: List[float]) -> Dict[str, Any]:
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(inputs_tensor).item()
        return {"prediccion": prediction, "accion": "comprar" if prediction > 0.5 else "vender"}

    async def run_analysis_loop(self):
        while True:
            if not self.circuit_breaker.check():
                await asyncio.sleep(60)
                continue

            exchange_data = await self.redis_client.get("exchange_data")
            macro_data = await self.redis_client.get("macro_data")

            if exchange_data and macro_data:
                exchange_json = json.loads(exchange_data)
                macro_json = json.loads(macro_data)
                inputs = [exchange_json["volume"], exchange_json["price_change"], macro_json["vix"], macro_json["dxy_change"]]
                inputs += [exchange_json.get(f"altcoin_{i}_volume", 0) for i in range(6)]

                if len(self.input_history) >= self.max_data_points:
                    self.input_history.pop(0)
                self.input_history.append(inputs)

                recommendation = await self.procesar_datos(inputs)
                recommendation["timestamp"] = datetime.utcnow().timestamp()

                datos_comprimidos = zstd.compress(json.dumps(recommendation).encode())
                mensaje = await serializar_mensaje(int(recommendation["timestamp"] % 1000000), self.canal, 0.0, True)

                await self.redis_client.xadd("crypto_trading_decisions", {"data": mensaje})

                self.logger.info(f"Recomendación IA generada: {recommendation}")

            await asyncio.sleep(60)

    async def detener(self):
        torch.save(self.model.state_dict(), self.model_path)
        self.logger.info("IAAnalisisProcessor detenido")
