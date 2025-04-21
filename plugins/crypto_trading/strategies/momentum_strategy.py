import logging
from typing import Dict, List
import numpy as np

class MomentumStrategy:
    def __init__(self, capital: float):
        self.logger = logging.getLogger("MomentumStrategy")
        self.capital = capital
        self.phase = self.get_phase()
        self.pending_signals = {}  # {exchange:pair: {"cycles": int, "sentiment": float}}

    def get_phase(self) -> int:
        """Determina la fase según el capital."""
        if self.capital < 1000:
            return 1
        elif self.capital < 10000:
            return 2
        elif self.capital < 100000:
            return 3
        elif self.capital < 1000000:
            return 4
        elif self.capital < 10000000:
            return 5
        else:
            return 6

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calcula el RSI (Relative Strength Index)."""
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
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: List[float]) -> tuple:
        """Calcula el MACD y la señal."""
        if len(prices) < 26:
            return 0.0, 0.0
        exp12 = np.convolve(prices, np.ones(12)/12, mode='valid')[-1]
        exp26 = np.convolve(prices, np.ones(26)/26, mode='valid')[-1]
        macd = exp12 - exp26
        signal = np.convolve(prices[-9:], np.ones(9)/9, mode='valid')[-1] if len(prices) >= 35 else 0.0
        return macd, signal

    def calculate_momentum(self, macro_data: Dict[str, float], crypto_data: Dict[str, float], prices: List[float], volatility: float) -> float:
        """Calcula el sentimiento del mercado con indicadores técnicos, ajustado por fase."""
        # Actualizar la fase según el capital
        self.phase = self.get_phase()

        # Umbrales de RSI y MACD según la fase
        phase_params = {
            1: {"rsi_buy": 75, "rsi_sell": 25, "sentiment_threshold": 50},
            2: {"rsi_buy": 72, "rsi_sell": 28, "sentiment_threshold": 50},
            3: {"rsi_buy": 70, "rsi_sell": 30, "sentiment_threshold": 60},
            4: {"rsi_buy": 68, "rsi_sell": 32, "sentiment_threshold": 60},
            5: {"rsi_buy": 65, "rsi_sell": 35, "sentiment_threshold": 70},
            6: {"rsi_buy": 60, "rsi_sell": 40, "sentiment_threshold": 70}
        }
        params = phase_params[self.phase]

        # Sentimiento base
        sentiment = (macro_data["sp500"] + macro_data["nasdaq"] - macro_data["dxy"] + macro_data["gold"] + macro_data["oil"]) * 100
        sentiment += (crypto_data["volume"] / 1000000) * 0.1

        # Ajustar con RSI
        rsi = self.calculate_rsi(prices)
        if rsi > params["rsi_buy"]:
            sentiment -= 50
        elif rsi < params["rsi_sell"]:
            sentiment += 50

        # Ajustar con MACD
        macd, signal = self.calculate_macd(prices)
        if macd > signal:
            sentiment += 25
        elif macd < signal:
            sentiment -= 25

        self.logger.info(f"[MomentumStrategy] Sentimiento calculado: {sentiment} (Phase: {self.phase}, RSI: {rsi}, MACD: {macd}, Signal: {signal})")
        return sentiment

    def decide_trade(self, exchange: str, pair: str, sentiment: float, volatility: float) -> str:
        """Decide si comprar o vender, con ciclos dinámicos ajustados por volatilidad."""
        # Actualizar la fase
        self.phase = self.get_phase()

        # Parámetros de ciclos según la fase
        phase_params = {
            1: {"cycles_needed_base": 2},
            2: {"cycles_needed_base": 2},
            3: {"cycles_needed_base": 1},
            4: {"cycles_needed_base": 1},
            5: {"cycles_needed_base": 1},
            6: {"cycles_needed_base": 1}
        }
        params = phase_params[self.phase]

        # Ajustar ciclos dinámicamente según volatilidad
        volatility_factor = max(volatility / 0.01, 1.0)  # Normalizar volatilidad
        cycles_needed = max(1, int(params["cycles_needed_base"] / volatility_factor))

        trade_id = f"{exchange}:{pair}"
        if trade_id not in self.pending_signals:
            self.pending_signals[trade_id] = {"cycles": 0, "sentiment": sentiment}
        else:
            self.pending_signals[trade_id]["cycles"] += 1
            self.pending_signals[trade_id]["sentiment"] = sentiment

        # Decidir si se confirma la operación
        if self.pending_signals[trade_id]["cycles"] >= cycles_needed:
            threshold = 50 if self.phase <= 2 else 60 if self.phase <= 4 else 70
            if sentiment > threshold:
                decision = "buy"
            elif sentiment < -threshold:
                decision = "sell"
            else:
                decision = "hold"
            del self.pending_signals[trade_id]
            self.logger.info(f"[MomentumStrategy] Decisión confirmada para {trade_id}: {decision} (Cycles: {cycles_needed})")
            return decision
        else:
            self.logger.info(f"[MomentumStrategy] Señal pendiente para {trade_id}: {self.pending_signals[trade_id]['cycles']}/{cycles_needed} ciclos")
            return "pending"

    def update_capital(self, new_capital: float):
        """Actualiza el capital y la fase."""
        self.capital = new_capital
        self.phase = self.get_phase()
        self.logger.info(f"[MomentumStrategy] Capital actualizado: ${self.capital}, Fase: {self.phase}")
