import logging
import random
import asyncio
from typing import Dict, Any
from plugins.crypto_trading.utils.helpers import CircuitBreaker

class ExecutionProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("ExecutionProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )
        self.active_capital = config["capital"]
        self.open_trades = config["open_trades"]
        self.num_exchanges = config["num_exchanges"]
        self.slippage_tolerance = 0.01  # Tolerancia al deslizamiento del 1%
        self.slippage_history = []  # Historial de deslizamiento para análisis

    def update_capital(self, capital: float):
        self.active_capital = capital
        self.logger.info(f"[ExecutionProcessor] Capital activo actualizado: ${self.active_capital}")

    def reset_active_capital(self):
        self.active_capital = 0
        self.logger.info("[ExecutionProcessor] Capital activo reiniciado")

    def calculate_slippage(self, volatility: float, volume: float) -> float:
        """Calcula el deslizamiento basado en la volatilidad y el volumen."""
        # El deslizamiento es mayor con alta volatilidad y bajo volumen
        base_slippage = volatility * 0.1  # Por ejemplo, 0.05 de volatilidad -> 0.005 (0.5%)
        volume_factor = max(0.1, 1000000 / volume)  # Menor volumen -> mayor deslizamiento
        slippage = base_slippage * volume_factor
        # Añadir un componente aleatorio para simular variaciones del mercado
        slippage *= random.uniform(0.8, 1.2)
        return slippage

    async def ejecutar_operacion(self, exchange: str, params: Dict[str, Any], paper_mode: bool = True, trade_multiplier: int = 1):
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo ejecución de operación")
            yield {"status": "skipped", "motivo": "circuito_abierto"}
            return

        try:
            # Obtener datos de mercado para calcular el deslizamiento
            market_data = await self.redis.get("market_data")
            if not market_data:
                self.logger.warning("No hay datos de mercado disponibles para calcular deslizamiento")
                yield {"status": "error", "motivo": "no_data"}
                return
            market_data = json.loads(market_data)
            crypto_data = market_data["crypto"]
            symbol = params["activo"]
            pair_data = crypto_data.get(symbol, {"volume": 1000000, "volatility": 0.01})
            volatility = pair_data.get("volatility", 0.01)
            volume = pair_data.get("volume", 1000000)

            # Calcular deslizamiento
            slippage = self.calculate_slippage(volatility, volume)
            expected_price = params["precio"]
            actual_price = expected_price * (1 + slippage if params["tipo"] == "buy" else 1 - slippage)
            slippage_percent = abs((actual_price - expected_price) / expected_price)

            # Verificar tolerancia al deslizamiento
            if slippage_percent > self.slippage_tolerance:
                self.logger.warning(f"Deslizamiento excede tolerancia: {slippage_percent*100:.2f}% (tolerancia: {self.slippage_tolerance*100:.2f}%)")
                yield {"status": "error", "motivo": f"Deslizamiento excede tolerancia: {slippage_percent*100:.2f}%"}
                return

            self.slippage_history.append(slippage_percent)
            self.logger.info(f"Deslizamiento aplicado: {slippage_percent*100:.2f}% (Precio esperado: ${expected_price}, Precio real: ${actual_price})")

            # Dividir la operación en fragmentos si es grande
            cantidad_total = params["cantidad"] * trade_multiplier
            fragment_size = max(0.01, cantidad_total / 3)  # Dividir en 3 fragmentos
            fragments = [fragment_size] * 3
            if cantidad_total % fragment_size != 0:
                fragments[-1] += cantidad_total % fragment_size

            for i, fragment in enumerate(fragments):
                if fragment == 0:
                    continue
                self.logger.info(f"Ejecutando fragmento {i+1}/{len(fragments)}: {fragment} unidades")

                orden_id = f"orden_{exchange}_{params['activo']}_{random.randint(1000, 9999)}"
                resultado = {
                    "orden_id": orden_id,
                    "exchange": exchange,
                    "activo": params["activo"],
                    "tipo": params["tipo"],
                    "precio": actual_price,
                    "cantidad": fragment,
                    "take_profit": actual_price * 1.03 if params["tipo"] == "buy" else actual_price * 0.97,
                    "stop_loss": actual_price * 0.97 if params["tipo"] == "buy" else actual_price * 1.03,
                    "timestamp": datetime.utcnow().isoformat(),
                    "close_timestamp": None,
                    "status": "open"
                }

                if not paper_mode:
                    self.logger.warning("Modo real no implementado, ejecutando en modo paper")
                self.active_capital += fragment
                yield {"status": "success", "result": resultado}

                # Simular latencia entre fragmentos
                await asyncio.sleep(random.uniform(0.5, 1.0))

        except Exception as e:
            self.logger.error(f"Error al ejecutar operación: {e}")
            self.cb.register_failure()
            yield {"status": "error", "motivo": str(e)}
