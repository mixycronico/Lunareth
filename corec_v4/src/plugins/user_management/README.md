# Plugin user_management para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que gestiona autenticación, autorización y perfiles de usuarios en el sistema de trading. Soporta registro, inicio de sesión con JWT, roles (administrador, miembro), y seguimiento de contribuciones/retiros, integrándose con `capital_pool` y `trading_execution`. Publica reportes en `user_data` y almacena datos en `user_db`.

## Propósito
Proveer un sistema seguro y escalable para gestionar usuarios, asegurando que solo usuarios autenticados puedan interactuar con el pool de capital y acceder a resultados de trading.

## Dependencias
- Python 3.8+
- psycopg2-binary==2.9.9
- zstandard==0.22.0
- bcrypt==4.1.3
- pyjwt==2.8.0

Instalar con:
```bash
pip install psycopg2-binary==2.9.9 zstandard==0.22.0 bcrypt==4.1.3 pyjwt==2.8.0
Estructura
  •	plugin.json: Metadatos del plugin.
  •	processors/user_processor.py: Lógica de autenticación y gestión de usuarios.
  •	utils/db.py: Gestión de la base de datos user_db.
  •	utils/auth.py: Funciones de autenticación con bcrypt y JWT.
  •	configs/plugins/user_management/user_management.yaml: Configuración del plugin.
  •	configs/plugins/user_management/schema.sql: Esquema de la base de datos.
  •	tests/plugins/test_user_management.py: Pruebas unitarias.
Configuración
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/user_management/processors
mkdir -p src/plugins/user_management/utils
mkdir -p configs/plugins/user_management
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade user_db al archivo docker-compose.yml:
services:
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
volumes:
  user_db-data:
3. Inicializar `user_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/user_management/schema.sql corec_v4-user_db-1:/schema.sql
docker exec corec_v4-user_db-1 psql -U user_management_user -d user_db -f /schema.sql
4. Configurar JWT Secret
Edita user_management.yaml, reemplazando jwt_secret con un valor seguro (ej., generado con openssl rand -hex 32).
5. Integrar en `main.py`
Añade la entidad para user_management en main.py:
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
Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Verificar Operaciones
El plugin gestiona registro, login, contribuciones/retiros, y reportes de trading, publicando en user_data y actualizando user_db.
Consulta los datos:
docker exec -it corec_v4-user_db-1 psql -U user_management_user -d user_db -c "SELECT * FROM users;"
3. Simular Datos (para pruebas)
Simula eventos para probar:
# Simular registro
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('user_data', '{\"action\": \"register\", \"data\": {\"email\": \"test@example.com\", \"password\": \"password123\", \"name\": \"Test User\", \"role\": \"member\"}}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# Simular contribución
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('capital_data', '{\"user_id\": \"user_test\", \"action\": \"contribute\", \"amount\": 500.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# Simular resultado de trading
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('trading_results', '{\"data\": \"$(echo '{\"user_id\": \"user_test\", \"profit\": 50.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
4. Ejecutar Pruebas
Valida el plugin:
pytest tests/plugins/test_user_management.py
5. Revisar Logs
Verifica la salida:
docker logs corec_v4-corec1-1
Funcionalidades
  •	Autenticación: Registro e inicio de sesión con JWT (expiración: 24 horas).
  •	Autorización: Roles (administrador, miembro) para restringir acciones.
  •	Gestión de Perfiles: Almacena datos de usuarios y actividades.
  •	Integración: Publica contribuciones/retiros en capital_data, asocia resultados de trading_results.
  •	Eficiencia: Caché de sesiones en Redis (TTL: 1800 segundos).
  •	Resiliencia: Circuit breakers, validación robusta.
Integración con Otros Plugins
  •	capital_pool: Publica contribuciones/retiros en capital_data.
  •	trading_execution: Asocia resultados de trading_results a usuarios.
  •	macro_sync: No depende directamente, pero puede usar macro_data para reportes.
  •	daily_settlement (futuro): Consolidará resultados por usuario.
Extensión
  •	Roles: Añade más roles (ej., auditor) en user_management.yaml.
  •	Autenticación: Integra OAuth2 o 2FA para mayor seguridad.
  •	Reportes: Amplía user_data con estadísticas detalladas (ROI por usuario).
  •	Interfaz: Desarrolla una CLI para interactuar con usuarios (Etapa 2).
Notas
  •	Plug-and-play: Independiente, usa canales para comunicación.
  •	Base de Datos: Inicializa user_db antes de usar.
  •	JWT Secret: Usa un secreto seguro en user_management.yaml.
  •	Contacto: Consulta al arquitecto principal para dudas sobre CoreC v4.
Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.
---

### Paso 4: Configuración del Entorno

#### 1. **Actualizar `docker-compose.yml`**
Añade `user_db`:

```yaml
services:
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
volumes:
  user_db-data:
2. Inicializar `user_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/user_management/schema.sql corec_v4-user_db-1:/schema.sql
docker exec corec_v4-user_db-1 psql -U user_management_user -d user_db -f /schema.sql
3. Instalar Dependencias
Instala las dependencias:
pip install psycopg2-binary==2.9.9 zstandard==0.22.0 bcrypt==4.1.3 pyjwt==2.8.0
4. Actualizar `main.py`
Añade la entidad para user_management:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
Punto de entrada para CoreC v4, registra entidades para los plugins predictor_temporal, market_monitor, exchange_sync, macro_sync, trading_execution, capital_pool, y user_management.
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
  2	Simular Datos: Asegúrate de que predictor_temporal, market_monitor, exchange_sync, macro_sync, trading_execution, y capital_pool estén generando datos. Simula eventos para user_management:
# Simular registro
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('user_data', '{\"action\": \"register\", \"data\": {\"email\": \"test@example.com\", \"password\": \"password123\", \"name\": \"Test User\", \"role\": \"member\"}}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# Simular contribución
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('capital_data', '{\"user_id\": \"user_test\", \"action\": \"contribute\", \"amount\": 500.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# Simular resultado de trading
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('trading_results', '{\"data\": \"$(echo '{\"user_id\": \"user_test\", \"profit\": 50.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  3	Verificar Resultados: Consulta user_db:
docker exec -it corec_v4-user_db-1 psql -U user_management_user -d user_db -c "SELECT * FROM users;"
  4	Ejecutar Pruebas: Valida el plugin:
pytest tests/plugins/test_user_management.py
  5	Revisar Logs: Verifica la salida:
docker logs corec_v4-corec1-1

Notas Importantes
  •	Integración con capital_pool: Asegúrate de que capital_pool procese eventos de capital_data generados por user_management. Verifica que capital_pool.yaml incluya capital_data en sus canales.
  •	Integración con trading_execution: Actualiza trading_execution para asociar órdenes a usuarios: async def execute_trading(self):
  •	    if "user_id" in self.price_cache[symbol]:
  •	        order["user_id"] = self.price_cache[symbol]["user_id"]
  •	
  •	Seguridad: Reemplaza jwt_secret en user_management.yaml con un valor seguro (ej., openssl rand -hex 32).
  •	OpenRouter: Mantenido en CoreCNucleus, listo para system_analyzer.

