import logging
from typing import Dict, List
import numpy as np

class MomentumStrategy:
    def __init__(self, capital: float):
        self.logger = logging.getLogger("MomentumStrategy")
        self.capital = capital
        self.phase = self.get_phase()
        self.pending_signals = {}
        self.is_uptrend = False  # Detectar subidas
        self.trade_multiplier = 1  # Multiplicador para más operaciones durante subidas

    def get_phase(self) -> int:
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
        if len(prices) < 26:
            return 0.0, 0.0
        exp12 = np.convolve(prices, np.ones(12)/12, mode='valid')[-1]
        exp26 = np.convolve(prices, np.ones(26)/26, mode='valid')[-1]
        macd = exp12 - exp26
        signal = np.convolve(prices[-9:], np.ones(9)/9, mode='valid')[-1] if len(prices) >= 35 else 0.0
        return macd, signal

    def calculate_momentum(self, macro_data: Dict[str, float], crypto_data: Dict[str, float], prices: List[float], volatility: float) -> float:
        self.phase = self.get_phase()

        phase_params = {
            1: {"rsi_buy": 75, "rsi_sell": 25, "sentiment_threshold": 50},
            2: {"rsi_buy": 72, "rsi_sell": 28, "sentiment_threshold": 50},
            3: {"rsi_buy": 70, "rsi_sell": 30, "sentiment_threshold": 60},
            4: {"rsi_buy": 68, "rsi_sell": 32, "sentiment_threshold": 60},
            5: {"rsi_buy": 65, "rsi_sell": 35, "sentiment_threshold": 70},
            6: {"rsi_buy": 60, "rsi_sell": 40, "sentiment_threshold": 70}
        }
        params = phase_params[self.phase]

        sentiment = (macro_data["sp500"] + macro_data["nasdaq"] - macro_data["dxy"] + macro_data["gold"] + macro_data["oil"]) * 100
        sentiment += (crypto_data["volume"] / 1000000) * 0.1

        rsi = self.calculate_rsi(prices)
        if rsi > params["rsi_buy"]:
            sentiment -= 50
        elif rsi < params["rsi_sell"]:
            sentiment += 50

        macd, signal = self.calculate_macd(prices)
        if macd > signal:
            sentiment += 25
        elif macd < signal:
            sentiment -= 25

        # Detectar subidas (tendencia alcista)
        price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
        rsi_trend = (self.calculate_rsi(prices[-10:]) - self.calculate_rsi(prices[-20:-10])) if len(prices) >= 20 else 0
        self.is_uptrend = price_change > 0.02 and rsi_trend > 0 and macd > signal  # Subida: +2%, RSI creciente, MACD alcista
        self.trade_multiplier = 2 if self.is_uptrend else 1  # Doblar operaciones durante subidas

        self.logger.info(f"[MomentumStrategy] Sentimiento calculado: {sentiment} (Phase: {self.phase}, RSI: {rsi}, MACD: {macd}, Signal: {signal}, Uptrend: {self.is_uptrend})")
        return sentiment

    def decide_trade(self, exchange: str, pair: str, sentiment: float, volatility: float) -> str:
        self.phase = self.get_phase()

        phase_params = {
            1: {"cycles_needed_base": 2},
            2: {"cycles_needed_base": 2},
            3: {"cycles_needed_base": 1},
            4: {"cycles_needed_base": 1},
            5: {"cycles_needed_base": 1},
            6: {"cycles_needed_base": 1}
        }
        params = phase_params[self.phase]

        volatility_factor = max(volatility / 0.01, 1.0)
        cycles_needed = max(1, int(params["cycles_needed_base"] / volatility_factor))

        trade_id = f"{exchange}:{pair}"
        if trade_id not in self.pending_signals:
            self.pending_signals[trade_id] = {"cycles": 0, "sentiment": sentiment}
        else:
            self.pending_signals[trade_id]["cycles"] += 1
            self.pending_signals[trade_id]["sentiment"] = sentiment

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

    def get_trade_multiplier(self) -> int:
        """Devuelve el multiplicador de operaciones para subidas."""
        return self.trade_multiplier

    def update_capital(self, new_capital: float):
        self.capital = new_capital
        self.phase = self.get_phase()
        self.logger.info(f"[MomentumStrategy] Capital actualizado: ${self.capital}, Fase: {self.phase}")
