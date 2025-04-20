import logging
from typing import List, Dict, Any
from corec.core import ComponenteBase

class ExchangeProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], nucleus):
        self.config = config
        self.nucleus = nucleus
        self.logger = logging.getLogger("ExchangeProcessor")
        self.exchanges = self.config.get("exchanges", [])

    async def detectar_disponibles(self) -> List[Dict[str, Any]]:
        """
        Detecta exchanges activos y filtra los que no responden.
        """
        disponibles = []
        for ex in self.exchanges:
            try:
                nombre = ex["name"]
                client = await self.nucleus.exchange_clients.get(nombre)
                await client.ping()
                disponibles.append({"name": nombre, "client": client})
            except Exception as e:
                self.logger.warning(f"Exchange {ex['name']} no disponible: {e}")
        self.logger.info(f"Exchanges activos: {[e['name'] for e in disponibles]}")
        return disponibles

    async def asignar_capital(self, total_capital: float) -> Dict[str, float]:
        """
        Distribuye capital entre exchanges disponibles dinámicamente.
        """
        activos = await self.detectar_disponibles()
        capital_por_exchange = {}
        if not activos:
            self.logger.error("No hay exchanges disponibles.")
            return capital_por_exchange

        capital_total_asignado = total_capital * 0.60  # 60% del capital total
        por_exchange = capital_total_asignado / len(activos)

        for ex in activos:
            capital_por_exchange[ex["name"]] = por_exchange

        self.logger.info(f"Distribución de capital: {capital_por_exchange}")
        return capital_por_exchange

    async def obtener_top_activos(self, client) -> List[str]:
        """
        Retorna los top 10 activos por volumen + BTC y ETH.
        """
        try:
            tickers = await client.fetch_tickers()
            ordenados = sorted(
                tickers.items(), key=lambda x: float(x[1].get("quoteVolume", 0)), reverse=True
            )
            top10 = [k for k, _ in ordenados if "/" in k][:10]
            activos = list({"BTC/USDT", "ETH/USDT"}.union(top10))
            self.logger.info(f"Activos seleccionados para operar: {activos}")
            return activos
        except Exception as e:
            self.logger.error(f"Error al obtener top activos: {e}")
            return []
