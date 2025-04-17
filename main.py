#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
Punto de entrada para CoreC v4, registra entidades para los plugins predictor_temporal, market_monitor, exchange_sync, macro_sync, trading_execution, capital_pool, user_management, daily_settlement, alert_manager, cli_manager, y system_analyzer.
"""

import asyncio
import os
from src.core.nucleus import CoreCNucleus
from src.core.celu_entidad import CeluEntidadCoreC

async def main():
    # Configuración inicial
    instance_id = os.getenv("INSTANCE_ID", "corec1")
    config_path = f"configs/core/corec_config_{instance_id}.json"
    nucleus = CoreCNucleus(config_path=config_path, instance_id=instance_id)

    # Registrar entidad para predictor_temporal
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_predictor_{instance_id}",
            nucleus.get_procesador("predictor_temporal"),
            "predictor_temporal",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para market_monitor
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_monitor_{instance_id}",
            nucleus.get_procesador("market_data"),
            "market_data",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para exchange_sync
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_exchange_{instance_id}",
            nucleus.get_procesador("exchange_data"),
            "exchange_data",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para macro_sync
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_macro_{instance_id}",
            nucleus.get_procesador("macro_data"),
            "macro_data",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para trading_execution
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_execution_{instance_id}",
            nucleus.get_procesador("trading_execution"),
            "trading_execution",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para capital_pool
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_capital_{instance_id}",
            nucleus.get_procesador("capital_data"),
            "capital_data",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para user_management
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_user_{instance_id}",
            nucleus.get_procesador("user_data"),
            "user_data",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para daily_settlement
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_settlement_{instance_id}",
            nucleus.get_procesador("settlement_data"),
            "settlement_data",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para alert_manager
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_alert_{instance_id}",
            nucleus.get_procesador("alert_data"),
            "alert_data",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para cli_manager
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_cli_{instance_id}",
            nucleus.get_procesador("cli_data"),
            "cli_data",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para system_analyzer
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_analyzer_{instance_id}",
            nucleus.get_procesador("system_insights"),
            "system_insights",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Registrar entidad para trading_results
    await nucleus.registrar_celu_entidad(
        CeluEntidadCoreC(
            f"nano_results_{instance_id}",
            nucleus.get_procesador("default"),
            "trading_results",
            5.0,
            nucleus.db_config,
            instance_id=instance_id
        )
    )

    # Iniciar el núcleo
    await nucleus.iniciar()

if __name__ == "__main__":
    asyncio.run(main())