import logging
from typing import List, Dict, Any
from corec.core import ComponenteBase
import ccxt.async_support as ccxt

class ExchangeProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], nucleus):
        self.config = config
        self.nucleus = nucleus
        self.logger = logging.getLogger("ExchangeProcessor")
        self.exchanges = self.config.get("exchanges", [])
        self.exchange_clients = {}

    async def inicializar(self):
        """Inicializa los clientes de los exchanges."""
        for ex in self.exchanges:
            try:
                nombre = ex["name"]
                exchange_class = getattr(ccxt, nombre)
                self.exchange_clients[nombre] = exchange_class({
                    "apiKey": ex["api_key"],
                    "secret": ex["api_secret"],
                    "enableRateLimit": True,
                })
            except Exception as e:
                self.logger.warning(f"No se pudo inicializar cliente para {ex['name']}: {e}")

    async def detectar_disponibles(self) -> List[Dict[str, Any]]:
        """Detecta exchanges activos y filtra los que no responden."""
        disponibles = []
        for ex in self.exchanges:
            try:
                nombre = ex["name"]
                client = self.exchange_clients.get(nombre)
                if client:
                    await client.fetch_status()
                    disponibles.append({"name": nombre, "client": client})
            except Exception as e:
                self.logger.warning(f"Exchange {ex['name']} no disponible: {e}")
        self.logger.info(f"Exchanges activos: {[e['name'] for e in disponibles]}")
        return disponibles

    async def asignar_capital(self, total_capital: float) -> Dict[str, float]:
        """Distribuye capital entre exchanges disponibles dinámicamente."""
        activos = await self.detectar_disponibles()
        capital_por_exchange = {}
        if not activos:
            self.logger.error("No hay exchanges disponibles.")
            return capital_por_exchange

        capital_total_asignado = total_capital * 0.60
        por_exchange = capital_total_asignado / len(activos)

        for ex in activos:
            capital_por_exchange[ex["name"]] = por_exchange

        self.logger.info(f"Distribución de capital: {capital_por_exchange}")
        return capital_por_exchange

    async def obtener_top_activos(self, client) -> List[str]:
        """Retorna BTC, ETH y los 10 activos con mayor volumen por exchange."""
        try:
            tickers = await client.fetch_tickers()
            ordenados = sorted(
                tickers.items(), key=lambda x: float(x[1].get("quoteVolume", 0)), reverse=True
            )
            top_activos = [k for k, _ in ordenados if "/" in k][:12]  # Top 12 para excluir BTC y ETH
            activos = ["BTC/USDT", "ETH/USDT"]
            for activo in top_activos:
                if activo not in ["BTC/USDT", "ETH/USDT"]:
                    activos.append(activo)
                if len(activos) >= 12:  # 2 (BTC, ETH) + 10 altcoins
                    break
            self.logger.info(f"Activos seleccionados para operar en {client.id}: {activos}")
            return activos
        except Exception as e:
            self.logger.error(f"Error al obtener top activos: {e}")
            return ["BTC/USDT", "ETH/USDT"]

    async def close(self):
        """Cierra todas las conexiones a los exchanges."""
        for client in self.exchange_clients.values():
            await client.close()
