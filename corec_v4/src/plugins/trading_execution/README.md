# Plugin trading_execution para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que ejecuta operaciones de trading (spot y futuros) en cinco exchanges (Binance, KuCoin, Bybit, OKX, Kraken), basado en predicciones, precios en tiempo real, y datos macroeconómicos. Implementa una estrategia de momentum con indicadores técnicos (RSI, MACD), ajustada por macro datos (S&P 500, VIX). Publica resultados en `trading_results` y almacena órdenes en `execution_db`.

## Propósito
Automatizar decisiones y ejecución de trading, reemplazando `EntidadExchangeManager` y `EntidadSyncStrategy` del sistema original, integrando predicciones de `predictor_temporal`, precios de `market_monitor`, datos macro de `macro_sync`, y estados de órdenes de `exchange_sync`.

## Dependencias
- Python 3.8+
- aiohttp==3.9.5
- psycopg2-binary==2.9.9
- zstandard==0.22.0
- backoff==2.2.1
- numpy==1.26.4
- ta==0.11.0

Instalar con:
```bash
pip install aiohttp==3.9.5 psycopg2-binary==2.9.9 zstandard==0.22.0 backoff==2.2.1 numpy==1.26.4 ta==0.11.0
Estructura
  •	plugin.json: Metadatos del plugin.
  •	processors/execution_processor.py: Lógica de trading y ejecución.
  •	utils/db.py: Gestión de la base de datos execution_db.
  •	configs/plugins/trading_execution/trading_execution.yaml: Configuración del plugin.
  •	configs/plugins/trading_execution/schema.sql: Esquema de la base de datos.
  •	tests/plugins/test_trading_execution.py: Pruebas unitarias.
Configuración
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/trading_execution/processors
mkdir -p src/plugins/trading_execution/utils
mkdir -p configs/plugins/trading_execution
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade execution_db al archivo docker-compose.yml:
services:
  execution_db:
    image: postgres:15
    environment:
      POSTGRES_DB: execution_db
      POSTGRES_USER: execution_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - execution_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  execution_db-data:
3. Inicializar `execution_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/trading_execution/schema.sql corec_v4-execution_db-1:/schema.sql
docker exec corec_v4-execution_db-1 psql -U execution_user -d execution_db -f /schema.sql
4. Configurar Claves de API
Edita trading_execution.yaml con las claves reales:
  •	Binance: api.binance.com
  •	KuCoin: www.kucoin.com
  •	Bybit: www.bybit.com
  •	OKX: www.okx.com
  •	Kraken: www.kraken.com
5. Integrar en `main.py`
Añade la entidad para trading_execution en main.py:
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
Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Verificar Operaciones
El plugin ejecuta órdenes cada 5 minutos basado en predicciones, precios, y macro datos. Las órdenes se publican en exchange_data (para exchange_sync) y los resultados en trading_results.
Consulta las órdenes:
docker exec -it corec_v4-execution_db-1 psql -U execution_user -d execution_db -c "SELECT * FROM orders;"
3. Simular Datos (para pruebas)
Asegúrate de que predictor_temporal, market_monitor, exchange_sync, y macro_sync estén generando datos. Simula eventos si es necesario:
# market_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('market_data', '{\"data\": \"$(echo '{\"symbol\": \"BTC/USDT\", \"price\": 35000.0, \"timestamp\": 1234567890.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# corec_stream_corec1
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('corec_stream_corec1', '{\"data\": \"$(echo '{\"symbol\": \"BTC/USDT\", \"prediction\": 36000.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# macro_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"vix_price\": 18.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
4. Ejecutar Pruebas
Valida el plugin:
pytest tests/plugins/test_trading_execution.py
5. Revisar Logs
Verifica la salida:
docker logs corec_v4-corec1-1
Funcionalidades
  •	Estrategia Momentum: Compra si predicción alcista (precio predicho > actual + 0.1%), RSI < 70, MACD > señal; vende si bajista, RSI > 30, MACD < señal.
  •	Ajuste Macro: Reduce riesgo si VIX > 20.
  •	Ejecución: Órdenes en spot y futuros (10x apalancamiento), en el exchange con menor spread.
  •	Gestión: Take-profit (+5%), stop-loss (-2%), riesgo 2% por operación.
  •	Eficiencia: Caché local, circuit breakers por exchange, reintentos.
  •	Almacenamiento: Guarda órdenes en execution_db, publica en trading_results.
Integración con Otros Plugins
  •	predictor_temporal: Predicciones via corec_stream_corec1.
  •	market_monitor: Precios via market_data.
  •	exchange_sync: Órdenes y estados via exchange_data.
  •	macro_sync: Macro datos via macro_data.
  •	capital_pool (futuro): Gestionará capital.
  •	daily_settlement (futuro): Consolidará resultados.
Extensión
  •	Estrategias: Añade scalping o breakout en execute_trading.
  •	Riesgo: Ajusta risk_per_trade o añade stop-loss dinámico.
  •	Macro Datos: Usa oro, petróleo para ajustes.
  •	Métricas: Registra profit, drawdown en trading_results.
Notas
  •	Plug-and-play: Independiente, usa canales para comunicación.
  •	Claves de API: Configura claves reales en trading_execution.yaml.
  •	Órdenes: Simuladas en pruebas; adapta place_order a APIs reales.
  •	Contacto: Consulta al arquitecto principal para dudas sobre CoreC v4.
Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.
---

### Paso 4: Configuración del Entorno

#### 1. **Actualizar `docker-compose.yml`**
Añade `execution_db`:

```yaml
services:
  execution_db:
    image: postgres:15
    environment:
      POSTGRES_DB: execution_db
      POSTGRES_USER: execution_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - execution_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  execution_db-data:
2. Inicializar `execution_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/trading_execution/schema.sql corec_v4-execution_db-1:/schema.sql
docker exec corec_v4-execution_db-1 psql -U execution_user -d execution_db -f /schema.sql
3. Instalar Dependencias
Instala las dependencias:
pip install aiohttp==3.9.5 psycopg2-binary==2.9.9 zstandard==0.22.0 backoff==2.2.1 numpy==1.26.4 ta==0.11.0
4. Actualizar `main.py`
Añade la entidad para trading_execution:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
Punto de entrada para CoreC v4, registra entidades para los plugins predictor_temporal, market_monitor, exchange_sync, macro_sync, y trading_execution.
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

Paso 5: Probar el Plugin
  1	Iniciar CoreC v4:
./scripts/start.sh
  2	Configurar Claves de API:
  •	Actualiza trading_execution.yaml con las claves reales de Binance, KuCoin, Bybit, OKX, y Kraken.
  •	Si no tienes claves, usa datos simulados (ver paso 3).
  3	Simular Datos: Asegúrate de que predictor_temporal, market_monitor, exchange_sync, y macro_sync estén generando datos. Simula eventos adicionales:
# market_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('market_data', '{\"data\": \"$(echo '{\"symbol\": \"BTC/USDT\", \"price\": 35000.0, \"timestamp\": 1234567890.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# corec_stream_corec1
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('corec_stream_corec1', '{\"data\": \"$(echo '{\"symbol\": \"BTC/USDT\", \"prediction\": 36000.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# macro_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"vix_price\": 18.0, \"nasdaq_price\": 15000.0, \"gold_price\": 1800.0, \"oil_price\": 80.0, \"altcoins_volume\": 1000000.0, \"news_sentiment\": 0.7}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# exchange_data (orden simulada)
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('exchange_data', '{\"data\": \"$(echo '{\"exchange\": \"binance\", \"order_id\": \"test123\", \"symbol\": \"BTC/USDT\", \"market\": \"spot\", \"side\": \"buy\", \"quantity\": 0.001, \"price\": 35000.0, \"status\": \"open\", \"timestamp\": 1234567890.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  4	Verificar Resultados: Consulta execution_db:
docker exec -it corec_v4-execution_db-1 psql -U execution_user -d execution_db -c "SELECT * FROM orders;"
  5	Ejecutar Pruebas: Valida el plugin:
pytest tests/plugins/test_trading_execution.py
  6	Revisar Logs: Verifica la salida:
docker logs corec_v4-corec1-1

Notas Importantes
  •	Claves de API: Configura claves reales en trading_execution.yaml para ejecución auténtica. Usa datos simulados para pruebas.
  •	Órdenes Simuladas: place_order y close_order simulan respuestas; adapta a APIs reales según la documentación de cada exchange.
  •	Integración: Asegúrate de que los plugins previos (predictor_temporal, market_monitor, exchange_sync, macro_sync) estén funcionando.
  •	Capital: Usa 1000 USD fijo hasta integrar capital_pool.

