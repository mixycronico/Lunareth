# Plugin market_monitor para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que monitorea en tiempo real los precios de criptomonedas (BTC, ETH, y altcoins dinámicos) desde el canal `exchange_data` y publica datos optimizados en el canal `market_data`. Consume la lista de altcoins dinámicos (top por volumen) desde el canal `macro_data`, manteniendo un diseño plug-and-play sin dependencias explícitas. Almacena precios en una base de datos propia (`monitor_db`) para análisis y auditoría.

## Propósito
Proveer un monitoreo robusto y continuo de precios de activos para el sistema de trading modular, alimentando plugins como `predictor_temporal` (predicciones) y `trading_execution` (ejecución de trades). Reemplaza las entidades `EntidadBTCWatcher`, `EntidadETHWatcher`, `EntidadAltcoinWatcher`, y parte de `EntidadTradingMonitor` del sistema de trading original.

## Dependencias
- Python 3.8+
- psycopg2-binary==2.9.9
- zstandard==0.22.0

Instalar con:
```bash
pip install psycopg2-binary==2.9.9 zstandard==0.22.0
Estructura
  •	plugin.json: Metadatos del plugin.
  •	processors/monitor_processor.py: Lógica de monitoreo de precios.
  •	utils/db.py: Gestión de la base de datos monitor_db.
  •	configs/plugins/market_monitor/market_monitor.yaml: Configuración del plugin.
  •	configs/plugins/market_monitor/schema.sql: Esquema de la base de datos.
  •	tests/plugins/test_market_monitor.py: Pruebas unitarias.
Configuración
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/market_monitor/processors
mkdir -p src/plugins/market_monitor/utils
mkdir -p configs/plugins/market_monitor
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade monitor_db al archivo docker-compose.yml:
services:
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
volumes:
  monitor_db-data:
3. Inicializar `monitor_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/market_monitor/schema.sql corec_v4-monitor_db-1:/schema.sql
docker exec corec_v4-monitor_db-1 psql -U monitor_user -d monitor_db -f /schema.sql
4. Integrar en `main.py`
Añade la entidad para market_monitor en main.py:
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
Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Simular Datos
El plugin consume precios desde exchange_data (generados por exchange_sync) y altcoins desde macro_data (generados por macro_sync). Para pruebas iniciales, simula eventos manuales:
  •	Simular altcoins desde macro_data:
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"altcoins\": [\"SOL/USDT\", \"ADA/USDT\", \"XRP/USDT\"]}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  •	Simular precios desde exchange_data:
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('exchange_data', '{\"data\": \"$(echo '{\"symbol\": \"BTC/USDT\", \"price\": 35000.0, \"timestamp\": 1234567890.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
3. Verificar Resultados
Consulta los precios almacenados en monitor_db:
docker exec -it corec_v4-monitor_db-1 psql -U monitor_user -d monitor_db -c "SELECT * FROM market_data;"
4. Ejecutar Pruebas
Valida el plugin con:
pytest tests/plugins/test_market_monitor.py
5. Revisar Logs
Verifica la salida:
docker logs corec_v4-corec1-1
Funcionalidades
  •	Monitoreo en Tiempo Real: Procesa precios de BTC/USDT, ETH/USDT, y altcoins dinámicos cada 60 segundos (configurable).
  •	Consumo de Datos: Obtiene precios desde exchange_data (generados por exchange_sync) y altcoins desde macro_data (generados por macro_sync).
  •	Publicación: Publica datos comprimidos (zstandard) en market_data para otros plugins.
  •	Almacenamiento: Guarda precios en monitor_db con índices optimizados para consultas.
  •	Resiliencia: Incluye circuit breaker (3 fallos, 900 segundos de espera) y fallbacks para mantener altcoins previos si no hay datos nuevos.
Integración con Otros Plugins
  •	predictor_temporal: Puede consumir market_data para generar predicciones. Actualiza su configuración para escuchar market_data: channels:
  •	  - "predictor_temporal"
  •	  - "macro_data"
  •	  - "market_data"
  •	
  •	exchange_sync: (Pendiente de desarrollo) Proveerá precios de cinco exchanges en exchange_data.
  •	macro_sync: (Pendiente de desarrollo) Proveerá altcoins y macro datos (S&P 500, DXY, sentimiento de noticias) en macro_data.
Extensión
  •	Añadir Símbolos: Modifica symbols en market_monitor.yaml para incluir más activos fijos (ej., BNB/USDT).
  •	Frecuencia: Ajusta update_interval para monitoreo más frecuente (ej., 30 segundos).
  •	Métricas: Añade indicadores técnicos (RSI, MACD) en monitor_processor.py para enriquecer market_data.
  •	Integración: Conecta con trading_execution para alimentar estrategias de trading.
Notas
  •	Plug-and-play: El plugin es independiente, no nombra otros plugins, y usa canales (market_data, macro_data, exchange_data) para comunicación.
  •	Dependencias Externas: Requiere exchange_sync y macro_sync para datos completos. Usa eventos manuales para pruebas iniciales.
  •	APIs: No consulta APIs directamente; depende de exchange_sync para precios y macro_sync para altcoins y macro datos.
  •	Contacto: Consulta al arquitecto principal para dudas sobre CoreC v4 o integración con otros plugins.
Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.
---
