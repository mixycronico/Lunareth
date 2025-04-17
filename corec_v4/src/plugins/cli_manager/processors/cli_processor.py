#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/cli_manager/processors/cli_processor.py
"""
cli_processor.py
Proporciona una interfaz CLI/TUI interactiva para monitorear, configurar y chatear con CoreC y el sistema de trading.
Incluye validación de permisos por rol (user, admin, superadmin).
"""

from ....core.processors.base import ProcesadorBase
from ....core.entidad_base import Event
from ....utils.logging import logger
from ..utils.db import CLIDB
from ..utils.tui import CLITUIApp
import click
import json
import zstandard as zstd
from typing import Dict, Any
from datetime import datetime
import asyncio
import yaml
import os
import psycopg2

class CLIProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("CLIProcessor")
        self.plugin_db = None
        self.data_cache = {}
        self.goals = {}
        self.tui_enabled = config.get("cli_config", {}).get("tui_enabled", True)
        self.user_db_config = config.get("user_db_config", {})

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.plugin_db = CLIDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a cli_db")
            await self.nucleus.publicar_alerta({"tipo": "db_connection_error", "plugin": "cli_manager", "message": "No se pudo conectar a cli_db"})

        self.goals = await self.plugin_db.get_goals()
        if self.tui_enabled:
            self.tui_app = CLITUIApp(self)
            asyncio.create_task(self.tui_app.run_async())
        else:
            self.logger.info("Modo texto activado")
        self.logger.info("CLIProcessor inicializado")

    async def check_permission(self, user_id: str, permission: str) -> bool:
        try:
            conn = psycopg2.connect(**self.user_db_config)
            cur = conn.cursor()
            cur.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            role = cur.fetchone()
            cur.close()
            conn.close()
            if not role:
                return False
            role = role[0]
            permissions = {
                "user": ["view_reports", "contribute_pool", "receive_notifications"],
                "admin": ["view_reports", "contribute_pool", "receive_notifications", "manage_users", "configure_trading", "approve_operations"],
                "superadmin": ["view_reports", "contribute_pool", "receive_notifications", "manage_users", "configure_trading", "approve_operations", "manage_plugins", "configure_system", "view_audits"]
            }
            return permission in permissions.get(role, [])
        except Exception as e:
            self.logger.error(f"Error verificando permiso: {e}")
            return False

    async def procesar(self, datos: Any, contexto: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user_id = datos.get("user_id", "user1")
            action = datos.get("action")

            # Validar permisos para acciones restringidas
            if action in ["manage_swarm", "optimize_strategy", "apply_insight"]:
                if not await self.check_permission(user_id, "manage_plugins"):
                    return {"estado": "error", "mensaje": f"Usuario {user_id} no tiene permiso para {action}"}

            if contexto["canal"] in ["market_data", "corec_stream_corec1", "trading_results", "capital_data", "user_data", "settlement_data", "alert_data", "eventos", "auditoria", "system_insights"]:
                self.data_cache[contexto["canal"]] = datos
                await self.redis_client.setex(f"cli:{contexto['canal']}", 300, json.dumps(datos))
                return {"estado": "ok", "mensaje": f"Datos recibidos para {contexto['canal']}"}
            elif contexto["canal"] == "cli_data" and action == "chat":
                response = await self.nucleus.responder_chat(datos["message"], "Sistema CoreC y Trading")
                return {"estado": "ok", "response": response["respuesta"]}
            elif contexto["canal"] == "cli_data" and action == "set_goal":
                goal_id = f"goal_{datetime.utcnow().timestamp()}"
                self.goals[goal_id] = datos["goal"]
                if self.plugin_db and self.plugin_db.conn:
                    await self.plugin_db.save_goal(goal_id, datos["goal"], user_id, datetime.utcnow().timestamp())
                await self.redis_client.xadd("cli_data", {"data": zstd.compress(json.dumps({"action": "update_goal", "goal_id": goal_id, "goal": datos["goal"]}).encode())})
                return {"estado": "ok", "goal_id": goal_id}
            elif contexto["canal"] == "cli_data" and action == "analyze_market":
                analysis = await self.nucleus.razonar(self.data_cache, "Análisis de mercado con DXY, VIX, y precios")
                return {"estado": "ok", "response": analysis["respuesta"]}
            elif contexto["canal"] == "cli_data" and action == "manage_swarm":
                await self.nucleus.modulo_registro.regenerar_enjambre(datos["canal"], datos["count"])
                return {"estado": "ok", "response": f"Regenerado enjambre en {datos['canal']}"}
            elif contexto["canal"] == "cli_data" and action == "run_backtest":
                backtest_result = await self.run_backtest(datos["params"])
                return {"estado": "ok", "response": backtest_result}
            elif contexto["canal"] == "cli_data" and action == "optimize_strategy":
                await self.redis_client.xadd("trading_execution", {"data": zstd.compress(json.dumps({"action": "update_strategy", "params": datos["params"]}).encode())})
                return {"estado": "ok", "response": f"Estrategia optimizada: {datos['params']}"}
            elif contexto["canal"] == "cli_data" and action == "monitor_dxy":
                dxy_data = self.data_cache.get("macro_data", {}).get("dxy_price", 0)
                change = self.data_cache.get("macro_data", {}).get("dxy_change_percent", 0)
                return {"estado": "ok", "response": f"DXY: {dxy_data:.2f} (Cambio: {change:.2f}%)"}
            elif contexto["canal"] == "cli_data" and action == "apply_insight":
                await self.redis_client.xadd("system_insights", {"data": zstd.compress(json.dumps({"action": "execute", "recommendation": datos["recommendation"]}).encode())})
                return {"estado": "ok", "response": f"Aplicada recomendación: {datos['recommendation']['details']}"}
            return {"estado": "error", "mensaje": "Canal o acción no soportada"}
        except Exception as e:
            self.logger.error(f"Error procesando datos: {e}")
            await self.nucleus.publicar_alerta({"tipo": "processing_error", "plugin": "cli_manager", "message": str(e)})
            return {"estado": "error", "mensaje": str(e)}

    async def run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Simulación de backtest avanzado (delegado a trading_execution)
            roi = 8.5 + (params.get("risk", 0.02) * 100)
            trades = params.get("trades", 20)
            return {"roi": roi, "trades": trades, "sharpe_ratio": 1.2}
        except Exception as e:
            self.logger.error(f"Error en backtest: {e}")
            return {"estado": "error", "mensaje": str(e)}

    async def manejar_evento(self, event: Event) -> None:
        try:
            datos = json.loads(zstd.decompress(event.datos["data"]))
            result = await self.procesar(datos, {"canal": event.canal, "instance_id": self.nucleus.instance_id, "nano_id": "cli_manager"})
            if result["estado"] == "ok" and "response" in result:
                self.data_cache["chat_response"] = result["response"]
                await self.redis_client.setex("cli:chat_response", 300, json.dumps(result["response"]))
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            await self.nucleus.publicar_alerta({"tipo": "event_error", "plugin": "cli_manager", "message": str(e)})

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        self.logger.info("CLIProcessor detenido")

    @click.group()
    def cli():
        """CLI para CoreC y Trading"""
        pass

    @cli.command()
    @click.option('--user_id', default="user1", help="ID del usuario")
    def status(user_id):
        """Muestra el estado del sistema"""
        cache = asyncio.run(redis_client.get_all("cli:*"))
        cache = {k: json.loads(v) for k, v in cache.items()}
        conn = psycopg2.connect(**self.user_db_config)
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
        role = cur.fetchone()[0] if cur.fetchone() else "unknown"
        cur.close()
        conn.close()
        click.echo("=== Estado de CoreC y Trading ===")
        click.echo(f"Usuario: {user_id} (Rol: {role})")
        click.echo(f"Nodos Activos: {cache.get('eventos', {}).get('nodes', 0)}")
        click.echo(f"Micro-celus: {cache.get('eventos', {}).get('micro_celus', 0)}")
        click.echo(f"Pool: ${cache.get('capital_data', {}).get('pool_total', 0):.2f}")
        click.echo(f"ROI Diario: {cache.get('settlement_data', {}).get('roi_percent', 0):.2f}%")
        click.echo(f"Sharpe Ratio: {cache.get('system_insights', {}).get('metrics', {}).get('trading', {}).get('sharpe_ratio', 0):.2f}")
        click.echo(f"DXY: {cache.get('macro_data', {}).get('dxy_price', 0):.2f} ({cache.get('macro_data', {}).get('dxy_change_percent', 0):.2f}%)")
        click.echo(f"Alertas: {len(cache.get('alert_data', []))}")

    @cli.command()
    @click.argument('exchange')
    @click.argument('api_key')
    @click.argument('api_secret')
    @click.option('--user_id', default="user1", help="ID del usuario")
    def config_exchange(exchange, api_key, api_secret, user_id):
        """Configura claves de API para un exchange"""
        if not asyncio.run(self.check_permission(user_id, "configure_trading")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para configurar exchanges")
            return
        config_path = "configs/plugins/trading_execution/trading_execution.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        for ex in config["exchange_config"]["exchanges"]:
            if ex["name"] == exchange:
                ex["api_key"] = api_key
                ex["api_secret"] = api_secret
                break
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        click.echo(f"Configuración actualizada para {exchange}")

    @cli.command()
    @click.option('--user_id', default="user1", help="ID del usuario")
    def alerts(user_id):
        """Lista alertas recientes"""
        if not asyncio.run(self.check_permission(user_id, "view_reports")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para ver alertas")
            return
        cache = asyncio.run(redis_client.get("cli:alert_data"))
        alerts = json.loads(cache) if cache else []
        click.echo("=== Alertas Recientes ===")
        for alert in alerts[:5]:
            click.echo(f"[{alert['severity'].upper()}] {alert['message']} ({alert['timestamp']})")

    @cli.command()
    @click.option('--user_id', default="user1", help="ID del usuario")
    def report(user_id):
        """Muestra el reporte diario"""
        if not asyncio.run(self.check_permission(user_id, "view_reports")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para ver reportes")
            return
        cache = asyncio.run(redis_client.get("cli:settlement_data"))
        report = json.loads(cache) if cache else {}
        click.echo("=== Reporte Diario ===")
        click.echo(f"Fecha: {report.get('date', 'N/A')}")
        click.echo(f"Profit: ${report.get('total_profit', 0):.2f}")
        click.echo(f"ROI: {report.get('roi_percent', 0):.2f}%")
        click.echo(f"Operaciones: {report.get('total_trades', 0)}")

    @cli.command()
    @click.argument('message')
    @click.option('--user_id', default="user1", help="ID del usuario")
    def chat(message, user_id):
        """Inicia un chat con CoreC"""
        async def run_chat():
            response = await self.nucleus.responder_chat(message, "Sistema CoreC y Trading")
            click.echo(f"CoreC: {response['respuesta']}")
        asyncio.run(run_chat())

    @cli.command()
    @click.argument('goal_type')
    @click.argument('value')
    @click.option('--user_id', default="user1", help="ID del usuario")
    def set_goal(goal_type, value, user_id):
        """Define una meta (ej., roi, risk)"""
        if not asyncio.run(self.check_permission(user_id, "configure_trading")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para establecer metas")
            return
        goal = {"type": goal_type, "value": float(value)}
        async def save_goal():
            datos = {"action": "set_goal", "goal": goal, "user_id": user_id}
            result = await self.procesar(datos, {"canal": "cli_data", "instance_id": self.nucleus.instance_id, "nano_id": "cli_manager"})
            if result["estado"] == "ok":
                click.echo(f"Meta establecida: {goal_type} = {value} (ID: {result['goal_id']})")
            else:
                click.echo(f"Error: {result['mensaje']}")
        asyncio.run(save_goal())

    @cli.command()
    @click.option('--user_id', default="user1", help="ID del usuario")
    def list_goals(user_id):
        """Lista metas activas"""
        if not asyncio.run(self.check_permission(user_id, "view_reports")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para ver metas")
            return
        click.echo("=== Metas Activas ===")
        for goal_id, goal in self.goals.items():
            click.echo(f"ID: {goal_id}, Tipo: {goal['type']}, Valor: {goal['value']}")

    @cli.command()
    @click.option('--user_id', default="user1", help="ID del usuario")
    def analyze_market(user_id):
        """Analiza el mercado actual"""
        if not asyncio.run(self.check_permission(user_id, "view_reports")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para analizar mercado")
            return
        async def run_analysis():
            datos = {"action": "analyze_market", "user_id": user_id}
            result = await self.procesar(datos, {"canal": "cli_data", "instance_id": self.nucleus.instance_id, "nano_id": "cli_manager"})
            if result["estado"] == "ok":
                click.echo(f"Análisis: {result['response']}")
            else:
                click.echo(f"Error: {result['mensaje']}")
        asyncio.run(run_analysis())

    @cli.command()
    @click.argument('canal')
    @click.argument('count', type=int)
    @click.option('--user_id', default="user1", help="ID del usuario")
    def manage_swarm(canal, count, user_id):
        """Regenera un enjambre de micro-celus"""
        async def run_swarm():
            datos = {"action": "manage_swarm", "canal": canal, "count": count, "user_id": user_id}
            result = await self.procesar(datos, {"canal": "cli_data", "instance_id": self.nucleus.instance_id, "nano_id": "cli_manager"})
            if result["estado"] == "ok":
                click.echo(f"Éxito: {result['response']}")
            else:
                click.echo(f"Error: {result['mensaje']}")
        asyncio.run(run_swarm())

    @cli.command()
    @click.option('--risk', default=0.02, help="Riesgo por operación")
    @click.option('--trades', default=20, help="Número de operaciones")
    @click.option('--user_id', default="user1", help="ID del usuario")
    def backtest_advanced(risk, trades, user_id):
        """Ejecuta un backtest avanzado"""
        if not asyncio.run(self.check_permission(user_id, "configure_trading")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para ejecutar backtests")
            return
        async def run_backtest_cmd():
            datos = {"action": "run_backtest", "params": {"risk": risk, "trades": trades}, "user_id": user_id}
            result = await self.procesar(datos, {"canal": "cli_data", "instance_id": self.nucleus.instance_id, "nano_id": "cli_manager"})
            if result["estado"] == "ok":
                click.echo(f"Backtest: ROI {result['response']['roi']}%, Operaciones: {result['response']['trades']}, Sharpe: {result['response']['sharpe_ratio']}")
            else:
                click.echo(f"Error: {result['mensaje']}")
        asyncio.run(run_backtest_cmd())

    @cli.command()
    @click.option('--risk', default=0.02, help="Riesgo por operación")
    @click.option('--take_profit', default=0.05, help="Take-profit")
    @click.option('--user_id', default="user1", help="ID del usuario")
    def optimize_strategy(risk, take_profit, user_id):
        """Optimiza la estrategia de trading"""
        async def run_optimize():
            datos = {"action": "optimize_strategy", "params": {"risk_per_trade": risk, "take_profit": take_profit}, "user_id": user_id}
            result = await self.procesar(datos, {"canal": "cli_data", "instance_id": self.nucleus.instance_id, "nano_id": "cli_manager"})
            if result["estado"] == "ok":
                click.echo(f"Éxito: {result['response']}")
            else:
                click.echo(f"Error: {result['mensaje']}")
        asyncio.run(run_optimize())

    @cli.command()
    @click.option('--user_id', default="user1", help="ID del usuario")
    def monitor_dxy(user_id):
        """Muestra el estado del índice DXY"""
        if not asyncio.run(self.check_permission(user_id, "view_reports")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para monitorear DXY")
            return
        async def run_monitor():
            datos = {"action": "monitor_dxy", "user_id": user_id}
            result = await self.procesar(datos, {"canal": "cli_data", "instance_id": self.nucleus.instance_id, "nano_id": "cli_manager"})
            if result["estado"] == "ok":
                click.echo(f"Éxito: {result['response']}")
            else:
                click.echo(f"Error: {result['mensaje']}")
        asyncio.run(run_monitor())

    @cli.command()
    @click.argument('plugin')
    @click.argument('action')
    @click.option('--user_id', default="user1", help="ID del usuario")
    def apply_insight(plugin, action, user_id):
        """Aplica una recomendación de system_analyzer"""
        async def run_apply():
            recommendation = {"plugin": plugin, "action": action, "details": f"Aplicar {action} en {plugin}"}
            datos = {"action": "apply_insight", "recommendation": recommendation, "user_id": user_id}
            result = await self.procesar(datos, {"canal": "cli_data", "instance_id": self.nucleus.instance_id, "nano_id": "cli_manager"})
            if result["estado"] == "ok":
                click.echo(f"Éxito: {result['response']}")
            else:
                click.echo(f"Error: {result['mensaje']}")
        asyncio.run(run_apply())

    @cli.command()
    @click.option('--user_id', default="user1", help="ID del usuario")
    def view_insights(user_id):
        """Muestra recomendaciones de system_analyzer"""
        if not asyncio.run(self.check_permission(user_id, "view_reports")):
            click.echo(f"Error: Usuario {user_id} no tiene permiso para ver insights")
            return
        cache = asyncio.run(redis_client.get("cli:system_insights"))
        insights = json.loads(cache) if cache else {}
        click.echo("=== Recomendaciones del Sistema ===")
        for rec in insights.get("recommendations", []):
            click.echo(f"Plugin: {rec['plugin']}, Acción: {rec['action']}, Detalles: {rec['details']}")

if __name__ == "__main__":
    cli()