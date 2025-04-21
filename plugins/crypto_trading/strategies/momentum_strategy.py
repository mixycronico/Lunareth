import logging
from typing import Dict, List
import numpy as np

class MomentumStrategy:
    def __init__(self):
        self.logger = logging.getLogger("MomentumStrategy")

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

    def calculate_momentum(self, macro_data: Dict[str, float], crypto_data: Dict[str, float], prices: List[float]) -> float:
        """Calcula el sentimiento del mercado con indicadores técnicos."""
        sentiment = (macro_data["sp500"] + macro_data["nasdaq"] - macro_data["dxy"] + macro_data["gold"] + macro_data["oil"]) * 100
        sentiment += (crypto_data["volume"] / 1000000) * 0.1

        rsi = self.calculate_rsi(prices)
        if rsi > 70:
            sentiment -= 50
        elif rsi < 30:
            sentiment += 50

        macd, signal = self.calculate_macd(prices)
        if macd > signal:
            sentiment += 25
        elif macd < signal:
            sentiment -= 25

        self.logger.info(f"[MomentumStrategy] Sentimiento calculado: {sentiment} (RSI: {rsi}, MACD: {macd}, Signal: {signal})")
        return sentiment

    def decide_trade(self, sentiment: float) -> str:
        """Decide si comprar o vender basado en el sentimiento."""
        if sentiment > 50:
            return "buy"
        elif sentiment < -50:
            return "sell"
        return "hold"
