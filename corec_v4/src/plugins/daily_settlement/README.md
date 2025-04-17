# Plugin daily_settlement para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que consolida resultados diarios de trading, calcula ROI, distribuye ganancias proporcionalmente a usuarios, y genera reportes detallados. Se integra con `trading_execution`, `capital_pool`, `user_management`, y `macro_sync`, publicando en `settlement_data` y almacenando en `settlement_db`.

## Propósito
Asegurar la transparencia y auditoría del sistema de trading familiar, consolidando ganancias, pérdidas, y movimientos financieros diariamente, actualizando el pool de capital y generando reportes.

## Dependencias
- Python 3.8+
- psycopg2-binary==2.9.9
- zstandard==0.22.0

Instalar con:
```bash
pip install psycopg2-binary==2.9.9 zstandard==0.22.0
Estructura
  •	plugin.json: Metadatos del plugin.
  •	processors/settlement_processor.py: Lógica de consolidación y reportes.
  •	utils/db.py: Gestión de la base de datos settlement_db.
  •	configs/plugins/daily_settlement/daily_settlement.yaml: Configuración del plugin.
  •	configs/plugins/daily_settlement/schema.sql: Esquema de la base de datos.
  •	tests/plugins/test_daily_settlement.py: Pruebas unitarias.
Configuración
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/daily_settlement/processors
mkdir -p src/plugins/daily_settlement/utils
mkdir -p configs/plugins/daily_settlement
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade settlement_db al archivo docker-compose.yml:
services:
  settlement_db:
    image: postgres:15
    environment:
      POSTGRES_DB: settlement_db
      POSTGRES_USER: settlement_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - settlement_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  settlement_db-data:
3. Inicializar `settlement_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/daily_settlement/schema.sql corec_v4-settlement_db-1:/schema.sql
docker exec corec_v4-settlement_db-1 psql -U settlement_user -d settlement_db -f /schema.sql
4. Actualizar `corec1.depends_on`
Añade settlement_db a las dependencias de corec1 en docker-compose.yml:
depends_on:
  - redis
  - postgres
  - trading_db
  - predictor_db
  - monitor_db
  - exchange_db
  - macro_db
  - execution_db
  - capital_db
  - user_db
  - settlement_db
5. Integrar en `main.py`
Añade la entidad para daily_settlement en main.py:
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
Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Verificar Reportes
El plugin consolida resultados diariamente a las 23:59 UTC, publicando en settlement_data y almacenando en settlement_db.
Consulta los reportes:
docker exec -it corec_v4-settlement_db-1 psql -U settlement_user -d settlement_db -c "SELECT * FROM reports;"
3. Simular Datos (para pruebas)
Simula eventos para probar:
# trading_results
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('trading_results', '{\"data\": \"$(echo '{\"profit\": 50.0, \"symbol\": \"BTC/USDT\", \"exchange\": \"binance\", \"user_id\": \"user1\"}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# capital_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('capital_data', '{\"data\": \"$(echo '{\"user_id\": \"user1\", \"action\": \"contribute\", \"amount\": 200.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# macro_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"vix_price\": 18.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# user_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('user_data', '{\"data\": \"$(echo '{\"user_id\": \"user1\", \"action\": \"register\", \"email\": \"test@example.com\"}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
4. Ejecutar Pruebas
Valida el plugin:
pytest tests/plugins/test_daily_settlement.py
5. Revisar Logs
Verifica la salida:
docker logs corec_v4-corec1-1
Funcionalidades
  •	Consolidación: Calcula ganancias, pérdidas, y ROI diario, distribuyendo proporcionalmente a usuarios.
  •	Reportes: Genera resúmenes con métricas (profit, ROI, operaciones, pool).
  •	Actualización: Sincroniza el pool, reinvirtiendo ganancias.
  •	Contexto Macro: Incluye datos macro (VIX, S&P 500) en reportes.
  •	Eficiencia: Caché en Redis (TTL: 86400 segundos).
  •	Resiliencia: Circuit breakers, fallbacks para datos faltantes.
Integración con Otros Plugins
  •	trading_execution: Consume trading_results para resultados de órdenes.
  •	capital_pool: Consume capital_data, actualiza el pool.
  •	user_management: Usa user_data para asociar actividades a usuarios.
  •	macro_sync: Usa macro_data para contexto.
  •	system_analyzer (futuro): Consumirá settlement_data para análisis.
Extensión
  •	Reportes: Añade métricas como drawdown o Sharpe ratio.
  •	Frecuencia: Permite consolidaciones intradiarias (micro-ciclos).
  •	Interfaz: Desarrolla una CLI para visualizar reportes (Etapa 2).
  •	Auditoría: Integra con OpenRouter en CoreCNucleus para análisis avanzado.
Notas
  •	Plug-and-play: Independiente, usa canales para comunicación.
  •	Base de Datos: Inicializa settlement_db antes de usar.
  •	Contacto: Consulta al arquitecto principal para dudas sobre CoreC v4.
Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.
---

### Paso 4: Configuración del Entorno

#### 1. **Actualizar `docker-compose.yml`**
Añade `settlement_db` y actualiza `corec1.depends_on`:

```yaml
version: '3.9'
services:
  corec1:
    build:
      context: .
      dockerfile: docker/Dockerfile
    environment:
      - ENVIRONMENT=development
      - INSTANCE_ID=corec1
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ./configs:/app/configs
      - ./src/plugins:/app/src/plugins
    depends_on:
      - redis
      - postgres
      - trading_db
      - predictor_db
      - monitor_db
      - exchange_db
      - macro_db
      - execution_db
      - capital_db
      - user_db
      - settlement_db
    ports:
      - "8000:8000"
    networks:
      - corec-network

  redis:
    image: redis:7.2
    networks:
      - corec-network

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: corec_db
      POSTGRES_USER: corec_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - corec-network

  trading_db:
    image: postgres:15
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - trading_db-data:/var/lib/postgresql/data
    networks:
      - corec-network

  predictor_db:
    image: postgres:15
    environment:
      POSTGRES_DB: predictor_db
      POSTGRES_USER: predictor_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - predictor_db-data:/var/lib/postgresql/data
    networks:
      - corec-network

  monitor_db:
    image: postgres:15
    environment:
      POSTGRES_DB: monitor_db
      POSTGRES_USER: monitor_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - monitor_db-data:/var/lib/postgresql/data
    networks:
      - corec-network

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

  capital_db:
    image: postgres:15
    environment:
      POSTGRES_DB: capital_db
      POSTGRES_USER: capital_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - capital_db-data:/var/lib/postgresql/data
    networks:
      - corec-network

  user_db:
    image: postgres:15
    environment:
      POSTGRES_DB: user_db
      POSTGRES_USER: user_management_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - user_db-data:/var/lib/postgresql/data
    networks:
      - corec-network

  settlement_db:
    image: postgres:15
    environment:
      POSTGRES_DB: settlement_db
      POSTGRES_USER: settlement_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - settlement_db-data:/var/lib/postgresql/data
    networks:
      - corec-network

networks:
  corec-network:
    driver: bridge

volumes:
  postgres-data:
  trading_db-data:
  predictor_db-data:
  monitor_db-data:
  exchange_db-data:
  macro_db-data:
  execution_db-data:
  capital_db-data:
  user_db-data:
  settlement_db-data:
2. Inicializar `settlement_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/daily_settlement/schema.sql corec_v4-settlement_db-1:/schema.sql
docker exec corec_v4-settlement_db-1 psql -U settlement_user -d settlement_db -f /schema.sql
3. Instalar Dependencias
Instala las dependencias:
pip install psycopg2-binary==2.9.9 zstandard==0.22.0
4. Actualizar `main.py`
Añade la entidad para daily_settlement:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
Punto de entrada para CoreC v4, registra entidades para los plugins predictor_temporal, market_monitor, exchange_sync, macro_sync, trading_execution, capital_pool, user_management, y daily_settlement.
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
  2	Simular Datos: Asegúrate de que todos los plugins estén generando datos. Simula eventos para daily_settlement:
# trading_results
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('trading_results', '{\"data\": \"$(echo '{\"profit\": 50.0, \"symbol\": \"BTC/USDT\", \"exchange\": \"binance\", \"user_id\": \"user1\"}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# capital_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('capital_data', '{\"data\": \"$(echo '{\"user_id\": \"user1\", \"action\": \"contribute\", \"amount\": 200.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# macro_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"vix_price\": 18.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# user_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('user_data', '{\"data\": \"$(echo '{\"user_id\": \"user1\", \"action\": \"register\", \"email\": \"test@example.com\"}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  3	Verificar Resultados: Consulta settlement_db:
docker exec -it corec_v4-settlement_db-1 psql -U settlement_user -d settlement_db -c "SELECT * FROM reports;"
  4	Ejecutar Pruebas: Valida el plugin:
pytest tests/plugins/test_daily_settlement.py
  5	Revisar Logs: Verifica la salida:
docker logs corec_v4-corec1-1

Notas Importantes
  •	Integración con Otros Plugins:
  ◦	Asegúrate de que trading_execution publique user_id en trading_results para asociar órdenes a usuarios.
  ◦	Verifica que capital_pool escuche capital_data para actualizaciones del pool.
  ◦	Confirma que user_management gestione usuarios correctamente.
  •	OpenRouter: Mantenido en CoreCNucleus, listo para system_analyzer.
  •	Órdenes Simuladas: Usa claves reales en trading_execution.yaml para ejecución auténtica.

