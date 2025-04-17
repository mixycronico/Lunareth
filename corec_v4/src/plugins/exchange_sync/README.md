# Plugin exchange_sync para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que consulta precios (spot y futures) y monitorea operaciones abiertas en cinco exchanges (Binance, KuCoin, Bybit, OKX, Kraken), publicando datos en el canal `exchange_data`. Diseñado para eficiencia, usa rotación de consultas, caché en Redis, manejo de límites de tasa, y circuit breakers por exchange. Almacena datos en una base de datos propia (`exchange_db`) para auditoría.

## Propósito
Centralizar las consultas a las APIs de exchanges, proporcionando precios y estados de operaciones para plugins como `market_monitor` y `trading_execution`, reemplazando la funcionalidad de `EntidadExchangeManager` del sistema de trading original.

## Dependencias
- Python 3.8+
- aiohttp==3.9.5
- psycopg2-binary==2.9.9
- zstandard==0.22.0
- backoff==2.2.1

Instalar con:
```bash
pip install aiohttp==3.9.5 psycopg2-binary==2.9.9 zstandard==0.22.0 backoff==2.2.1
Estructura
  •	plugin.json: Metadatos del plugin.
  •	processors/exchange_processor.py: Lógica de consultas a exchanges.
  •	utils/db.py: Gestión de la base de datos exchange_db.
  •	configs/plugins/exchange_sync/exchange_sync.yaml: Configuración del plugin.
  •	configs/plugins/exchange_sync/schema.sql: Esquema de la base de datos.
  •	tests/plugins/test_exchange_sync.py: Pruebas unitarias.
Configuración
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/exchange_sync/processors
mkdir -p src/plugins/exchange_sync/utils
mkdir -p configs/plugins/exchange_sync
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade exchange_db al archivo docker-compose.yml:
services:
  exchange_db:
    image: postgres:15
    environment:
      POSTGRES_DB: exchange_db
      POSTGRES_USER: exchange_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - exchange_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  exchange_db-data:
3. Inicializar `exchange_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/exchange_sync/schema.sql corec_v4-exchange_db-1:/schema.sql
docker exec corec_v4-exchange_db-1 psql -U exchange_user -d exchange_db -f /schema.sql
4. Configurar Claves de API
Edita exchange_sync.yaml con las claves reales de cada exchange:
  •	Binance: api.binance.com
  •	KuCoin: www.kucoin.com
  •	Bybit: www.bybit.com
  •	OKX: www.okx.com
  •	Kraken: www.kraken.com
5. Integrar en `main.py`
Añade la entidad para exchange_sync en main.py:
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
Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Verificar Datos
El plugin consulta precios (spot y futures) cada 5 minutos por exchange (escalonados cada 1 minuto) y operaciones abiertas continuamente. Los datos se publican en exchange_data y se almacenan en exchange_db.
Consulta los datos:
docker exec -it corec_v4-exchange_db-1 psql -U exchange_user -d exchange_db -c "SELECT * FROM exchange_data;"
docker exec -it corec_v4-exchange_db-1 psql -U exchange_user -d exchange_db -c "SELECT * FROM open_orders;"
3. Simular Datos (para pruebas sin claves de API)
Si no tienes claves, simula eventos en exchange_data:
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('exchange_data', '{\"data\": \"$(echo '{\"exchange\": \"binance\", \"symbol\": \"BTCUSDT\", \"market\": \"spot\", \"price\": 35000.0, \"timestamp\": 1234567890.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
4. Ejecutar Pruebas
Valida el plugin con:
pytest tests/plugins/test_exchange_sync.py
5. Revisar Logs
Verifica la salida:
docker logs corec_v4-corec1-1
Funcionalidades
  •	Consulta de Precios: Obtiene precios de spot y futures para símbolos configurados (ej., BTC/USDT, ETH/USDT) cada 5 minutos por exchange.
  •	Monitoreo de Operaciones Abiertas: Consulta órdenes abiertas (spot y futures) continuamente.
  •	Eficiencia:
  ◦	Rotación: Consulta un exchange cada 60 segundos (5 minutos / 5 exchanges).
  ◦	Caché: Almacena precios en Redis (TTL: 1800 segundos).
  ◦	Límites de Tasa: Reintentos con backoff exponencial.
  ◦	Circuit Breakers: Pausa consultas por exchange tras 3 fallos (900 segundos).
  •	Publicación: Publica datos comprimidos (zstandard) en exchange_data.
  •	Almacenamiento: Guarda precios y órdenes en exchange_db.
Integración con Otros Plugins
  •	market_monitor: Consume exchange_data para monitorear precios y publicar en market_data.
  •	predictor_temporal: Puede usar market_data (vía market_monitor) para predicciones.
  •	trading_execution: (Pendiente de desarrollo) Usará exchange_data para ejecutar trades.
Extensión
  •	Añadir Exchanges: Agrega más exchanges en exchange_config.exchanges en exchange_sync.yaml.
  •	Símbolos: Amplía symbols por exchange (ej., SOL/USDT, XRP/USDT).
  •	Operaciones: Integra ejecución de trades (en trading_execution) usando las APIs autenticadas.
  •	Métricas: Añade volumen de trading o profundidad de mercado en fetch_exchange_data.
Notas
  •	Plug-and-play: El plugin es independiente, no nombra otros plugins, y usa el canal exchange_data para comunicación.
  •	Claves de API: Configura claves reales en exchange_sync.yaml. Usa datos simulados para pruebas iniciales.
  •	Personalización: Adapta fetch_spot_price, fetch_futures_price, y fetch_open_orders a las APIs específicas de cada exchange.
  •	Contacto: Consulta al arquitecto principal para dudas sobre CoreC v4 o integración.
Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.
---

### Paso 4: Configuración del Entorno

#### 1. **Actualizar `docker-compose.yml`**
Añade `exchange_db`:

```yaml
services:
  exchange_db:
    image: postgres:15
    environment:
      POSTGRES_DB: exchange_db
      POSTGRES_USER: exchange_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - exchange_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  exchange_db-data:
2. Inicializar `exchange_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/exchange_sync/schema.sql corec_v4-exchange_db-1:/schema.sql
docker exec corec_v4-exchange_db-1 psql -U exchange_user -d exchange_db -f /schema.sql
3. Instalar Dependencias
Instala las dependencias:
pip install aiohttp==3.9.5 psycopg2-binary==2.9.9 zstandard==0.22.0 backoff==2.2.1
4. Actualizar `main.py`
Añade la entidad para exchange_sync:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
Punto de entrada para CoreC v4, registra entidades para los plugins predictor_temporal, market_monitor y exchange_sync.
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

    # Iniciar el núcleo
    await nucleus.iniciar()

if __name__ == "__main__":
    asyncio.run(main())

Paso 5: Probar el Plugin
  1	Iniciar CoreC v4:
./scripts/start.sh
  2	Configurar Claves de API:
  •	Actualiza exchange_sync.yaml con las claves reales de Binance, KuCoin, Bybit, OKX, y Kraken.
  •	Si no tienes claves, usa datos simulados (ver paso 3).
  3	Simular Datos (para pruebas sin claves): Inserta un evento manual en exchange_data:
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('exchange_data', '{\"data\": \"$(echo '{\"exchange\": \"binance\", \"symbol\": \"BTCUSDT\", \"market\": \"spot\", \"price\": 35000.0, \"timestamp\": 1234567890.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  4	Verificar Resultados: Consulta exchange_db:
docker exec -it corec_v4-exchange_db-1 psql -U exchange_user -d exchange_db -c "SELECT * FROM exchange_data;"
docker exec -it corec_v4-exchange_db-1 psql -U exchange_user -d exchange_db -c "SELECT * FROM open_orders;"
  5	Ejecutar Pruebas: Valida el plugin:
pytest tests/plugins/test_exchange_sync.py
  6	Revisar Logs: Verifica la salida:
docker logs corec_v4-corec1-1

Notas Importantes
  •	Claves de API:
  ◦	Obtén claves de Binance, KuCoin, Bybit, OKX, y Kraken.
  ◦	Actualiza exchange_sync.yaml con las claves reales.
  ◦	Para pruebas, usa eventos simulados o modifica fetch_spot_price, fetch_futures_price, y fetch_open_orders con datos dummy.
  •	Personalización:
  ◦	Las URLs en fetch_spot_price, fetch_futures_price, y fetch_open_orders son genéricas. Adapta cada método a las APIs específicas:
  ▪	Binance: /api/v3/ticker/price, /fapi/v1/ticker/price.
  ▪	KuCoin: /api/v1/market/orderbook/level1.
  ▪	Bybit: /v2/public/tickers.
  ▪	OKX: /api/v5/market/ticker.
  ▪	Kraken: /0/public/Ticker.
  ◦	Añade autenticación (API key, secret) según la documentación de cada exchange.
  •	Integración:
  ◦	market_monitor ya está configurado para consumir exchange_data.
  ◦	Asegúrate de que predictor_temporal escuche market_data para usar los precios.

README para Desarrolladores
El README proporcionado arriba es completo y está listo para tu equipo. Asegúrate de incluirlo en src/plugins/exchange_sync/README.md.

