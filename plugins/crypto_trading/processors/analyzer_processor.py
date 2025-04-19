#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/analyzer_processor.py
Analiza métricas de CoreC y trading, proponiendo y ejecutando optimizaciones automáticamente.
Incluye Sharpe Ratio para evaluación de rendimiento ajustado al riesgo.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import json
import asyncio
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

class AnalyzerProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("AnalyzerProcessor")
        self.analysis_interval = self.config.get("analyzer_config", {}).get("analysis_interval", 300)
        self.auto_execute = self.config.get("analyzer_config", {}).get("auto_execute", True)
        self.circuit_breaker = CircuitBreaker(
            self.config.get("analyzer_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
            self.config.get("analyzer_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
        )
        self.plugin_db = TradingDB(self.config.get("db_config", {}))
        self.metrics_cache = {}

    async def inicializar(self):
        await self.plugin_db.connect()
        asyncio.create_task(self.analyze_system())
        self.logger.info("AnalyzerProcessor inicializado")

    async def execute_action(self, recommendation: Dict[str, Any]) -> bool:
        try:
            plugin = recommendation["plugin"]
            action = recommendation["action"]
            details = recommendation["details"]
            if plugin == "predictor_temporal" and action == "retrain_model":
                datos_comprimidos = zstd.compress(json.dumps({"action": "retrain"}).encode())
                await self.redis_client.xadd("crypto_trading_data", {"data": datos_comprimidos})
            elif plugin == "trading_execution" and action == "adjust_strategy":
                risk = 0.03 if "aumentar riesgo" in details.lower() else 0.01
                datos_comprimidos = zstd.compress(json.dumps({"action": "update_risk", "risk_per_trade": risk}).encode())
                await self.redis_client.xadd("crypto_trading_data", {"data": datos_comprimidos})
            elif plugin == "capital_pool" and action == "reduce_risk":
                datos_comprimidos = zstd.compress(json.dumps({"action": "update_phase", "phase": "conservative"}).encode())
                await self.redis_client.xadd("crypto_trading_data", {"data": datos_comprimidos})
            elif plugin == "corec" and action == "regenerate_swarm":
                await self.nucleus.modulo_registro.regenerar_enjambre("crypto_trading_data", 100)
            elif plugin == "alert_manager" and action == "adjust_thresholds":
                datos_comprimidos = zstd.compress(json.dumps({"action": "update_threshold", "vix_threshold": 22}).encode())
                await self.redis_client.xadd("crypto_trading_data", {"data": datos_comprimidos})
            self.logger.info(f"Ejecutada acción: {plugin} - {action} 🌟")
            return True
        except Exception as e:
            self.logger.error(f"Error ejecutando acción: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "action_error",
                "plugin": "crypto_trading",
                "message": str(e)
            })
            return False

    async def calculate_sharpe_ratio(self, profits: List[float]) -> float:
        if not profits or len(profits) < 2:
            return 0.0
        returns = np.array(profits) / 1000.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe

    async def analyze_system(self):
        while True:
            if not self.circuit_breaker.check():
                await asyncio.sleep(60)
                continue
            try:
                metrics = {
                    "predictor": {"mse": self.metrics_cache.get("predictor_temporal", {}).get("mse", 0)},
                    "trading": {
                        "roi": self.metrics_cache.get("settlement_data", {}).get("roi_percent", 0),
                        "trades": self.metrics_cache.get("trading_results", {}).get("total_trades", 0),
                        "profits": self.metrics_cache.get("trading_results", {}).get("profits", [])
                    },
                    "capital": {"pool_total": self.metrics_cache.get("capital_data", {}).get("pool_total", 0)},
                    "alerts": {
                        "count": len(self.metrics_cache.get("alert_data", [])),
                        "high_severity": sum(1 for a in self.metrics_cache.get("alert_data", []) if a["severity"] == "high")
                    },
                    "corec": {
                        "nodes": self.metrics_cache.get("eventos", {}).get("nodes", 0),
                        "load": self.metrics_cache.get("auditoria", {}).get("load", 0)
                    },
                    "macro": {
                        "vix": self.metrics_cache.get("macro_data", {}).get("vix_price", 0),
                        "dxy_change": self.metrics_cache.get("macro_data", {}).get("dxy_change_percent", 0)
                    }
                }
                metrics["trading"]["sharpe_ratio"] = await self.calculate_sharpe_ratio(metrics["trading"]["profits"])
                context = "Análisis del sistema CoreC y trading para optimización, incluyendo Sharpe Ratio"
                recommendations = []
                if metrics["predictor"]["mse"] > 15:
                    recommendations.append({"plugin": "predictor_temporal", "action": "retrain_model", "details": "MSE alto, reentrenar LSTM"})
                if metrics["trading"]["roi"] < 10:
                    recommendations.append({"plugin": "trading_execution", "action": "adjust_strategy", "details": "Aumentar riesgo a 3% o priorizar altcoins"})
                if metrics["trading"]["sharpe_ratio"] < 1.0:
                    recommendations.append({"plugin": "trading_execution", "action": "adjust_strategy", "details": "Sharpe Ratio bajo, optimizar riesgo-retorno"})
                if metrics["alerts"]["high_severity"] > 2:
                    recommendations.append({"plugin": "alert_manager", "action": "adjust_thresholds", "details": "Muchas alertas altas, aumentar umbral VIX"})
                if metrics["corec"]["load"] > 0.6:
                    recommendations.append({"plugin": "corec", "action": "regenerate_swarm", "details": "Carga alta, regenerar micro-celus"})
                if metrics["macro"]["dxy_change"] > 0.5:
                    recommendations.append({"plugin": "capital_pool", "action": "reduce_risk", "details": "DXY alto, reducir riesgo a 1%"})
                insight = {
                    "timestamp": datetime.utcnow().timestamp(),
                    "metrics": metrics,
                    "recommendations": recommendations,
                    "analysis": "Análisis completado con recomendaciones generadas"
                }
                datos_comprimidos = zstd.compress(json.dumps(insight).encode())
                mensaje = await serializar_mensaje(int(insight["timestamp"] % 1000000), self.canal, 0.0, True)
                await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                await self.plugin_db.save_insight(
                    timestamp=insight["timestamp"],
                    metrics=metrics,
                    recommendations=recommendations,
                    analysis=insight["analysis"]
                )
                if self.auto_execute:
                    for rec in recommendations:
                        executed = await self.execute_action(rec)
                        await self.nucleus.publicar_alerta({
                            "tipo": "action_executed" if executed else "action_failed",
                            "plugin": "crypto_trading",
                            "message": f"{'Ejecutada' if executed else 'Falló'} acción: {rec['plugin']} - {rec['action']}"
                        })
                self.logger.info(f"Análisis completado: {len(recommendations)} recomendaciones generadas")
            except Exception as e:
                self.logger.error(f"Error en análisis: {e}")
                self.circuit_breaker.register_failure()
                await self.nucleus.publicar_alerta({
                    "tipo": "analysis_error",
                    "plugin": "crypto_trading",
                    "message": str(e)
                })
            await asyncio.sleep(self.analysis_interval)

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            self.metrics_cache[mensaje["tipo"]] = mensaje
            await self.redis_client.setex(f"analyzer:{mensaje['tipo']}", 300, json.dumps(mensaje))
            self.logger.debug(f"Métrica recibida: {mensaje['tipo']}")
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            self.circuit_breaker.register_failure()

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("AnalyzerProcessor detenido")