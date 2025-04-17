# Plugin capital_pool para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que gestiona un pool de capital compartido para usuarios en el sistema de trading. Soporta contribuciones, retiros, reinversión de ganancias, y asignación de fondos para operaciones, con fases dinámicas basadas en el nivel de capital ($100-$999, $1,000-$9,999, $10,000+). Publica asignaciones en `capital_data` para `trading_execution`, consume resultados de `trading_results`, y ajusta el riesgo con datos macro de `macro_data`. Almacena datos en `capital_db`.

## Propósito
Centralizar la gestión de capital, asegurando que no más del 60% del pool esté activo en trades, con ajustes automáticos por fase y reinversión de ganancias, reemplazando la funcionalidad de gestión de capital del sistema original.

## Dependencias
- Python 3.8+
- psycopg2-binary==2.9.9
- zstandard==0.22.0

Instalar con:
```bash
pip install psycopg2-binary==2.9.9 zstandard==0.22.0
Estructura
  •	plugin.json: Metadatos del plugin.
  •	processors/capital_processor.py: Lógica de gestión de capital.
  •	utils/db.py: Gestión de la base de datos capital_db.
  •	configs/plugins/capital_pool/capital_pool.yaml: Configuración del plugin.
  •	configs/plugins/capital_pool/schema.sql: Esquema de la base de datos.
  •	tests/plugins/test_capital_pool.py: Pruebas unitarias.
Configuración
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/capital_pool/processors
mkdir -p src/plugins/capital_pool/utils
mkdir -p configs/plugins/capital_pool
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade capital_db al archivo docker-compose.yml:
services:
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
volumes:
  capital_db-data:
3. Inicializar `capital_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/capital_pool/schema.sql corec_v4-capital_db-1:/schema.sql
docker exec corec_v4-capital_db-1 psql -U capital_user -d capital_db -f /schema.sql
4. Integrar en `main.py`
Añade la entidad para capital_pool en main.py:
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
Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Verificar Operaciones
El plugin gestiona contribuciones, retiros, y asignaciones cada 5 minutos, publicando en capital_data y actualizando el pool con trading_results.
Consulta los datos:
docker exec -it corec_v4-capital_db-1 psql -U capital_user -d capital_db -c "SELECT * FROM contributions;"
docker exec -it corec_v4-capital_db-1 psql -U capital_user -d capital_db -c "SELECT * FROM withdrawals;"
docker exec -it corec_v4-capital_db-1 psql -U capital_user -d capital_db -c "SELECT * FROM pool_state;"
3. Simular Datos (para pruebas)
Simula eventos para probar:
# trading_results
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('trading_results', '{\"data\": \"$(echo '{\"profit\": 50.0, \"quantity\": 0.001, \"price\": 35000.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# macro_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"vix_price\": 18.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
4. Ejecutar Pruebas
Valida el plugin:
pytest tests/plugins/test_capital_pool.py
5. Revisar Logs
Verifica la salida:
docker logs corec_v4-corec1-1
Funcionalidades
  •	Gestión de Capital: Maneja contribuciones (mínimo $100), retiros, y reinversión de ganancias.
  •	Fases Dinámicas: Ajusta riesgo según el capital ($100-$999: 3%, $1,000-$9,999: 2%, $10,000+: 1%).
  •	Asignación: Limita capital activo a 60%, asigna fondos por operación.
  •	Ajuste Macro: Reduce riesgo si VIX > 20.
  •	Eficiencia: Caché en Redis (TTL: 1800 segundos), consultas optimizadas.
  •	Resiliencia: Circuit breakers, fallbacks para macro datos.
Integración con Otros Plugins
  •	trading_execution: Consume capital_data para asignar fondos; publica trading_results para actualizar el pool.
  •	macro_sync: Ajusta riesgo con macro_data.
  •	user_management (futuro): Gestionará autenticación de usuarios.
  •	daily_settlement (futuro): Consolidará resultados diarios.
Extensión
  •	Fases: Añade más fases en capital_config.phases.
  •	Riesgo: Ajusta risk_per_trade o usa oro/petróleo para ajustes macro.
  •	Usuarios: Integra con user_management para autenticación.
  •	Métricas: Registra ROI por usuario en capital_db.
Notas
  •	Plug-and-play: Independiente, usa canales para comunicación.
  •	Base de Datos: Asegúrate de inicializar capital_db antes de usar.
  •	Contacto: Consulta al arquitecto principal para dudas sobre CoreC v4.
Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.
---

### Paso 4: Configuración del Entorno

#### 1. **Actualizar `docker-compose.yml`**
Añade `capital_db`:

```yaml
services:
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
volumes:
  capital_db-data:
2. Inicializar `capital_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/capital_pool/schema.sql corec_v4-capital_db-1:/schema.sql
docker exec corec_v4-capital_db-1 psql -U capital_user -d capital_db -f /schema.sql
3. Instalar Dependencias
Instala las dependencias:
pip install psycopg2-binary==2.9.9 zstandard==0.22.0
4. Actualizar `main.py`
Añade la entidad para capital_pool:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
Punto de entrada para CoreC v4, registra entidades para los plugins predictor_temporal, market_monitor, exchange_sync, macro_sync, trading_execution, y capital_pool.
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
  2	Simular Datos: Asegúrate de que predictor_temporal, market_monitor, exchange_sync, macro_sync, y trading_execution estén generando datos. Simula eventos para capital_pool:
# Simular contribución
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('capital_data', '{\"user_id\": \"user1\", \"action\": \"contribute\", \"amount\": 500.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# Simular retiro
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('capital_data', '{\"user_id\": \"user1\", \"action\": \"withdraw\", \"amount\": 100.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# Simular resultado de trading
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('trading_results', '{\"data\": \"$(echo '{\"profit\": 50.0, \"quantity\": 0.001, \"price\": 35000.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# Simular macro datos
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"vix_price\": 18.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  3	Verificar Resultados: Consulta capital_db:
docker exec -it corec_v4-capital_db-1 psql -U capital_user -d capital_db -c "SELECT * FROM contributions;"
docker exec -it corec_v4-capital_db-1 psql -U capital_user -d capital_db -c "SELECT * FROM withdrawals;"
docker exec -it corec_v4-capital_db-1 psql -U capital_user -d capital_db -c "SELECT * FROM pool_state;"
  4	Ejecutar Pruebas: Valida el plugin:
pytest tests/plugins/test_capital_pool.py
  5	Revisar Logs: Verifica la salida:
docker logs corec_v4-corec1-1

Notas Importantes
  •	Integración con trading_execution: Asegúrate de que trading_execution escuche capital_data para usar las asignaciones. Actualiza trading_execution.yaml: channels:
  •	  - "market_data"
  •	  - "macro_data"
  •	  - "exchange_data"
  •	  - "corec_stream_corec1"
  •	  - "trading_results"
  •	  - "capital_data"  # Añadido
  •	
  •	Claves de API: No se requieren para capital_pool, pero verifica las claves en exchange_sync.yaml y trading_execution.yaml para ejecución real.
  •	Usuarios: El plugin asume IDs de usuario simples (ej., “user1”). Integra con user_management (futuro) para autenticación.
  •	OpenRouter: Mantenido en CoreCNucleus, listo para system_analyzer.

