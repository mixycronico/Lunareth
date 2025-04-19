#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/settlement_processor.py
Consolida resultados diarios de trading, genera reportes, y actualiza el pool de capital.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta

class SettlementProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("SettlementProcessor")
        self.settlement_time = self.config.get("settlement_config", {}).get("settlement_time", "23:59")
        self.circuit_breaker = CircuitBreaker(
            self.config.get("settlement_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
            self.config.get("settlement_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
        )
        self.plugin_db = TradingDB(self.config.get("db_config", {}))
        self.trading_results = []
        self.capital_movements = []
        self.macro_context = {}
        self.user_data = {}

    async def inicializar(self):
        await self.plugin_db.connect()
        asyncio.create_task(self.daily_settlement())
        self.logger.info("SettlementProcessor inicializado")

    async def consolidate_results(self) -> Dict[str, Any]:
        if not self.circuit_breaker.check():
            return {"status": "error", "message": "Circuit breaker activo"}
        try:
            total_profit = sum(result.get("profit", 0) for result in self.trading_results)
            total_trades = len(self.trading_results)
            trades_by_symbol = {}
            trades_by_exchange = {}
            for result in self.trading_results:
                symbol = result.get("symbol", "unknown")
                exchange = result.get("exchange", "unknown")
                trades_by_symbol[symbol] = trades_by_symbol.get(symbol, 0) + 1
                trades_by_exchange[exchange] = trades_by_exchange.get(exchange, 0) + 1
            pool_total = await self.plugin_db.get_pool_total()
            active_capital = await self.plugin_db.get_active_capital()
            phase = "unknown"
            for p in self.config.get("capital_config", {}).get("phases", []):
                if p["min"] <= pool_total < p["max"]:
                    phase = p["name"]
            roi = (total_profit / pool_total) * 100 if pool_total > 0 else 0
            user_distributions = {}
            users = await self.plugin_db.get_users()
            for user_id, contribution in users.items():
                proportion = contribution / pool_total if pool_total > 0 else 0
                user_distributions[user_id] = total_profit * proportion
            report = {
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "total_profit": total_profit,
                "roi_percent": roi,
                "total_trades": total_trades,
                "trades_by_symbol": trades_by_symbol,
                "trades_by_exchange": trades_by_exchange,
                "pool_total": pool_total,
                "active_capital": active_capital,
                "phase": phase,
                "user_distributions": user_distributions,
                "macro_context": self.macro_context,
                "timestamp": datetime.utcnow().timestamp()
            }
            await self.plugin_db.save_report(
                date=report["date"],
                total_profit=total_profit,
                roi_percent=roi,
                total_trades=total_trades,
                report_data=report,
                timestamp=report["timestamp"]
            )
            datos_comprimidos = zstd.compress(json.dumps(report).encode())
            mensaje = await serializar_mensaje(int(report["timestamp"] % 1000000), self.canal, total_profit, True)
            await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
            await self.redis_client.xadd("capital_data", {
                "data": zstd.compress(json.dumps({
                    "action": "update",
                    "profit": total_profit,
                    "timestamp": datetime.utcnow().timestamp()
                }).encode())
            })
            self.logger.info(f"Reporte diario generado: {report['date']}, ROI: {roi:.2f}% 🌟")
            return {"status": "ok", "report": report}
        except Exception as e:
            self.logger.error(f"Error consolidando resultados: {e}")
            self.circuit_breaker.register_failure()
            return {"status": "error", "message": str(e)}

    async def daily_settlement(self):
        while True:
            now = datetime.utcnow()
            settlement_time = datetime.strptime(self.settlement_time, "%H:%M").replace(
                year=now.year, month=now.month, day=now.day
            )
            if now >= settlement_time:
                await self.consolidate_results()
                self.trading_results = []
                self.capital_movements = []
                next_settlement = settlement_time + timedelta(days=1)
                await asyncio.sleep((next_settlement - now).total_seconds())
            else:
                await asyncio.sleep(60)

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "trading_results":
                self.trading_results.append(mensaje)
                self.logger.debug(f"Resultado de trading recibido: {mensaje['profit']}")
            elif mensaje.get("tipo") == "capital_data":
                self.capital_movements.append(mensaje)
                self.logger.debug(f"Movimiento de capital recibido: {mensaje['action']}")
            elif mensaje.get("tipo") == "macro_data":
                self.macro_context = mensaje
                self.logger.info(f"Datos macro recibidos: {self.macro_context}")
            elif mensaje.get("tipo") == "user_data":
                user_id = mensaje.get("user_id")
                self.user_data[user_id] = mensaje
                self.logger.debug(f"Evento de usuario recibido para {user_id}")
        except Exception as e:
            self.logger.error(f"Error procesando evento: {e}")
            self.circuit_breaker.register_failure()

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("SettlementProcessor detenido")