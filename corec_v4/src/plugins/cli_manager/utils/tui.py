#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/cli_manager/utils/tui.py
"""
tui.py
Interfaz TUI para el plugin cli_manager usando Textual, con ventana de chat, panel de insights y DXY.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static, Input, TextLog
from textual.containers import Container, Vertical
import json
import asyncio
from datetime import datetime

class CLITUIApp(App):
    CSS = """
    Container {
        layout: grid;
        grid-size: 2 2;
        height: 100%;
    }
    Vertical {
        height: 100%;
    }
    DataTable {
        height: 33%;
        width: 100%;
    }
    TextLog {
        height: 33%;
        width: 100%;
        background: black;
        color: white;
    }
    Input {
        width: 100%;
    }
    Static {
        background: darkblue;
        color: white;
        padding: 1;
    }
    """

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            DataTable(id="corec_status", show_cursor=True),
            DataTable(id="trading_metrics", show_cursor=True),
            DataTable(id="alerts", show_cursor=True),
            Vertical(
                DataTable(id="insights", show_cursor=True),
                TextLog(id="chat_log"),
                Input(id="chat_input", placeholder="Escribe tu mensaje a CoreC..."),
                Static("Comandos: [s]tatus, [c]onfig, [a]lerts, [r]eport, [g]oals, [m]arket, [w]arm, [b]acktest, [i]nsights, [o]ptimize, [d]xy", id="commands")
            )
        )
        yield Footer()

    async def on_mount(self) -> None:
        self.update_tables()
        self.set_interval(5, self.update_tables)

    def update_tables(self):
        corec_table = self.query_one("#corec_status", DataTable)
        trading_table = self.query_one("#trading_metrics", DataTable)
        alerts_table = self.query_one("#alerts", DataTable)
        insights_table = self.query_one("#insights", DataTable)
        chat_log = self.query_one("#chat_log", TextLog)

        corec_table.clear(columns=True).add_columns("Métrica", "Valor")
        trading_table.clear(columns=True).add_columns("Métrica", "Valor")
        alerts_table.clear(columns=True).add_columns("Severidad", "Mensaje", "Hora")
        insights_table.clear(columns=True).add_columns("Plugin", "Acción", "Detalles")

        cache = asyncio.run(self.processor.redis_client.get_all("cli:*"))
        cache = {k: json.loads(v) for k, v in cache.items()}

        corec_table.add_rows([
            ("Nodos", cache.get("eventos", {}).get("nodes", 0)),
            ("Micro-celus", cache.get("eventos", {}).get("micro_celus", 0)),
            ("Carga", f"{cache.get('auditoria', {}).get('load', 0):.2f}%")
        ])

        trading_table.add_rows([
            ("Pool", f"${cache.get('capital_data', {}).get('pool_total', 0):.2f}"),
            ("ROI Diario", f"{cache.get('settlement_data', {}).get('roi_percent', 0):.2f}%"),
            ("Órdenes", cache.get('trading_results', {}).get('total_trades', 0)),
            ("BTC/USDT Predicción", f"${cache.get('corec_stream_corec1', {}).get('prediction', 0):.2f}"),
            ("DXY", f"{cache.get('macro_data', {}).get('dxy_price', 0):.2f} ({cache.get('macro_data', {}).get('dxy_change_percent', 0):.2f}%)")
        ])

        alerts = cache.get("alert_data", [])[:5]
        alerts_table.add_rows([
            (a["severity"], a["message"], datetime.fromtimestamp(a["timestamp"]).strftime("%H:%M:%S"))
            for a in alerts
        ])

        insights = cache.get("system_insights", {}).get("recommendations", [])[:5]
        insights_table.add_rows([
            (r["plugin"], r["action"], r["details"])
            for r in insights
        ])

        if "chat_response" in cache:
            chat_log.write(f"CoreC: {cache['chat_response']}\n")

    async def on_input_submitted(self, event):
        if event.input.id == "chat_input":
            message = event.value
            event.input.value = ""
            chat_log = self.query_one("#chat_log", TextLog)
            chat_log.write(f"Tú: {message}\n")
            async def send_chat():
                datos = {"action": "chat", "message": message}
                result = await self.processor.procesar(datos, {"canal": "cli_data", "instance_id": self.processor.nucleus.instance_id, "nano_id": "cli_manager"})
                if result["estado"] == "ok":
                    chat_log.write(f"CoreC: {result['response']}\n")
            asyncio.create_task(send_chat())

    def on_key(self, event):
        if event.key == "s":
            asyncio.create_task(self.run_command("status"))
        elif event.key == "c":
            asyncio.create_task(self.run_command("config_exchange"))
        elif event.key == "a":
            asyncio.create_task(self.run_command("alerts"))
        elif event.key == "r":
            asyncio.create_task(self.run_command("report"))
        elif event.key == "g":
            asyncio.create_task(self.run_command("list_goals"))
        elif event.key == "m":
            asyncio.create_task(self.run_command("analyze_market"))
        elif event.key == "w":
            asyncio.create_task(self.run_command("manage_swarm"))
        elif event.key == "b":
            asyncio.create_task(self.run_command("backtest_advanced"))
        elif event.key == "i":
            asyncio.create_task(self.run_command("view_insights"))
        elif event.key == "o":
            asyncio.create_task(self.run_command("optimize_strategy"))
        elif event.key == "d":
            asyncio.create_task(self.run_command("monitor_dxy"))

    async def run_command(self, command):
        self.processor.cli.commands[command].callback()