# Plugin macro_sync para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que centraliza la obtención de datos macroeconómicos y de mercado desde Alpha Vantage (S&P 500, Nasdaq, VIX, oro, petróleo), CoinMarketCap (top 10 altcoins, volumen), y CoinDesk (sentimiento de noticias cripto). Publica datos en el canal `macro_data` para plugins como `predictor_temporal`, `market_monitor`, y `trading_execution`. Opera cada 5 minutos durante 7:00-17:00 (hora de Nueva York), con caché en Redis, circuit breakers por API, y almacenamiento en `macro_db`.

## Propósito
Proveer un panorama macroeconómico completo, priorizando indicadores financieros (80% peso: S&P 500, Nasdaq, VIX, oro, petróleo), altcoins y volumen (15%), y sentimiento de noticias (5%), reemplazando `EntidadAlphaVantageSync` del sistema original.

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
  •	processors/macro_processor.py: Lógica de consultas a APIs.
  •	utils/db.py: Gestión de la base de datos macro_db.
  •	configs/plugins/macro_sync/macro_sync.yaml: Configuración del plugin.
  •	configs/plugins/macro_sync/schema.sql: Esquema de la base de datos.
  •	tests/plugins/test_macro_sync.py: Pruebas unitarias.
Configuración
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/macro_sync/processors
mkdir -p src/plugins/macro_sync/utils
mkdir -p configs/plugins/macro_sync
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade macro_db al archivo docker-compose.yml:
services:
  macro_db:
    image: postgres:15
    environment:
      POSTGRES_DB: macro_db
      POSTGRES_USER: macro_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - macro_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  macro_db-data:
3. Inicializar `macro_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/macro_sync/schema.sql corec_v4-macro_db-1:/schema.sql
docker exec corec_v4-macro_db-1 psql -U macro_user -d macro_db -f /schema.sql
4. Configurar Claves de API
Edita macro_sync.yaml con las claves reales:
  •	Alpha Vantage: www.alphavantage.co
  •	CoinMarketCap: coinmarketcap.com
  •	CoinDesk: www.coindesk.com (obtener API de noticias)
5. Integrar en `main.py`
Añade la entidad para macro_sync en main.py:
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
Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Verificar Datos
El plugin consulta APIs cada 5 minutos (7:00-17:00, hora de Nueva York) y publica en macro_data. Los datos se cachean en Redis (TTL: 1800 segundos) y se almacenan en macro_db.
Consulta los datos:
docker exec -it corec_v4-macro_db-1 psql -U macro_user -d macro_db -c "SELECT * FROM macro_metrics;"
3. Simular Datos (para pruebas sin claves)
Inserta un evento manual:
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"nasdaq_price\": 15000.0, \"vix_price\": 20.0, \"gold_price\": 1800.0, \"oil_price\": 80.0, \"altcoins\": [\"SOL/USDT\", \"ADA/USDT\"], \"altcoins_volume\": 1000000.0, \"news_sentiment\": 0.7, \"timestamp\": 1234567890.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
4. Ejecutar Pruebas
Valida el plugin:
pytest tests/plugins/test_macro_sync.py
5. Revisar Logs
Verifica la salida:
docker logs corec_v4-corec1-1
Funcionalidades
  •	Consulta de Datos:
  ◦	Alpha Vantage: S&P 500, Nasdaq, VIX, oro (XAU/USD), petróleo (WTI).
  ◦	CoinMarketCap: Top 10 altcoins, volumen total.
  ◦	CoinDesk: Sentimiento de noticias cripto.
  •	Prioridad: 80% indicadores macro, 15% altcoins, 5% sentimiento.
  •	Frecuencia: Cada 5 minutos, 7:00-17:00 (hora de Nueva York).
  •	Eficiencia:
  ◦	Consultas paralelas.
  ◦	Caché en Redis (TTL: 1800 segundos).
  ◦	Reintentos con backoff exponencial.
  ◦	Circuit breakers por API (3 fallos, 900 segundos).
  •	Publicación: Datos comprimidos (zstandard) en macro_data.
  •	Almacenamiento: Guarda datos en macro_db.
Integración con Otros Plugins
  •	predictor_temporal: Ajusta predicciones con S&P 500, VIX, etc.
  •	market_monitor: Usa altcoins dinámicos para monitoreo.
  •	trading_execution: (Futuro) Ajusta estrategias con macro datos.
  •	exchange_sync: Puede usar volumen de altcoins para priorizar exchanges.
Extensión
  •	Fuentes: Añade DXY o tasas de interés en fetch_alpha_vantage.
  •	Sentimiento: Mejora análisis con OpenRouter o más fuentes (Bloomberg).
  •	Frecuencia: Ajusta fetch_interval (ej., 3 minutos).
  •	Almacenamiento: Registra datos brutos de noticias para análisis.
Notas
  •	Plug-and-play: Independiente, usa macro_data para comunicación.
  •	Claves de API: Configura claves reales en macro_sync.yaml. Usa datos simulados para pruebas.
  •	CoinDesk API: La API de noticias puede requerir personalización; adapta fetch_coindesk según documentación.
  •	Contacto: Consulta al arquitecto principal para dudas sobre CoreC v4.
Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.
---

### Paso 4: Configuración del Entorno

#### 1. **Actualizar `docker-compose.yml`**
Añade `macro_db`:

```yaml
services:
  macro_db:
    image: postgres:15
    environment:
      POSTGRES_DB: macro_db
      POSTGRES_USER: macro_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - macro_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  macro_db-data:
2. Inicializar `macro_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/macro_sync/schema.sql corec_v4-macro_db-1:/schema.sql
docker exec corec_v4-macro_db-1 psql -U macro_user -d macro_db -f /schema.sql
3. Instalar Dependencias
Instala las dependencias:
pip install aiohttp==3.9.5 psycopg2-binary==2.9.9 zstandard==0.22.0 backoff==2.2.1
4. Actualizar `main.py`
Asegúrate de que main.py incluya la entidad para macro_sync:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
Punto de entrada para CoreC v4, registra entidades para los plugins predictor_temporal, market_monitor, exchange_sync y macro_sync.
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

    # Iniciar el núcleo
    await nucleus.iniciar()

if __name__ == "__main__":
    asyncio.run(main())

Paso 5: Probar el Plugin
  1	Iniciar CoreC v4:
./scripts/start.sh
  2	Configurar Claves de API:
  •	Actualiza macro_sync.yaml con las claves reales de:
  ◦	Alpha Vantage: www.alphavantage.co
  ◦	CoinMarketCap: coinmarketcap.com
  ◦	CoinDesk: Contacta a CoinDesk para su API de noticias o usa una alternativa (ej., NewsAPI con filtro para cripto).
  •	Si no tienes claves, usa datos simulados (ver paso 3).
  3	Simular Datos (para pruebas sin claves): Inserta un evento manual:
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"nasdaq_price\": 15000.0, \"vix_price\": 20.0, \"gold_price\": 1800.0, \"oil_price\": 80.0, \"altcoins\": [\"SOL/USDT\", \"ADA/USDT\", \"XRP/USDT\"], \"altcoins_volume\": 1000000.0, \"news_sentiment\": 0.7, \"timestamp\": 1234567890.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  4	Verificar Resultados: Consulta macro_db:
docker exec -it corec_v4-macro_db-1 psql -U macro_user -d macro_db -c "SELECT * FROM macro_metrics;"
  5	Ejecutar Pruebas: Valida el plugin:
pytest tests/plugins/test_macro_sync.py
  6	Revisar Logs: Verifica la salida:
docker logs corec_v4-corec1-1

Notas Importantes
  •	Claves de API:
  ◦	Reemplaza las claves en macro_sync.yaml con las reales.
  ◦	La API de CoinDesk para noticias puede no estar disponible directamente; en ese caso, modifica fetch_coindesk para usar NewsAPI con filtro para “CoinDesk” o una fuente similar.
  •	CoinDesk API:
  ◦	El código usa un placeholder (currentprice.json). Adapta fetch_coindesk según la documentación de CoinDesk o usa una API alternativa (ej., NewsAPI con q=cryptocurrency).
  •	Integración:
  ◦	predictor_temporal y market_monitor ya están configurados para consumir macro_data.
  ◦	Asegúrate de que trading_execution (próximo plugin) escuche macro_data para ajustes de estrategia.
  •	Eficiencia:
  ◦	Consultas escalonadas y paralelas evitan saturación.
  ◦	Caché en Redis y compresión optimizan el rendimiento.
  ◦	Circuit breakers por API aseguran resiliencia.
