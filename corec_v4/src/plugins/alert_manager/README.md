# Plugin alert_manager para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que centraliza la gestión de alertas y notificaciones para todo el sistema, recopilando eventos críticos de los plugins actuales (`predictor_temporal`, `market_monitor`, `exchange_sync`, `macro_sync`, `trading_execution`, `capital_pool`, `user_management`, `daily_settlement`) y futuros plugins. Clasifica alertas por severidad (crítica, alta, media, baja) y envía notificaciones a través de múltiples canales (logs, correo, Discord). Publica reportes en `alert_data` y almacena datos en `alert_db`.

## Propósito
Proveer un sistema de alertas generalizado, escalable, y plug-and-play que monitoree eventos críticos (errores, circuit breakers, pérdidas, movimientos financieros) y notifique a los administradores, asegurando transparencia y resiliencia en el sistema de trading familiar.

## Dependencias
- Python 3.8+
- psycopg2-binary==2.9.9
- zstandard==0.22.0
- aiohttp==3.9.5
- backoff==2.2.1

Instalar con:
```bash
pip install psycopg2-binary==2.9.9 zstandard==0.22.0 aiohttp==3.9.5 backoff==2.2.1
Estructura
    •	plugin.json: Metadatos del plugin.
    •	processors/alert_processor.py: Lógica de clasificación y notificación de alertas.
    •	utils/db.py: Gestión de la base de datos alert_db.
    •	utils/notify.py: Funciones para enviar notificaciones (correo, Discord).
    •	configs/plugins/alert_manager/alert_manager.yaml: Configuración del plugin.
    •	configs/plugins/alert_manager/schema.sql: Esquema de la base de datos.
    •	tests/plugins/test_alert_manager.py: Pruebas unitarias.
Configuración
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/alert_manager/processors
mkdir -p src/plugins/alert_manager/utils
mkdir -p configs/plugins/alert_manager
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade alert_db al archivo docker-compose.yml:
services:
  alert_db:
    image: postgres:15
    environment:
      POSTGRES_DB: alert_db
      POSTGRES_USER: alert_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - alert_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  alert_db-data:
Actualiza las dependencias de corec1:
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
  - alert_db
3. Inicializar `alert_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/alert_manager/schema.sql corec_v4-alert_db-1:/schema.sql
docker exec corec_v4-alert_db-1 psql -U alert_user -d alert_db -f /schema.sql
4. Configurar Notificaciones
Edita alert_manager.yaml con credenciales reales para notificaciones:
    •	Correo: Configura smtp_server, smtp_port, sender, password, y recipient (ej., SendGrid, Gmail).
    •	Discord: Habilita con un webhook_url válido.
5. Integrar en `main.py`
Añade la entidad para alert_manager en main.py:
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
Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Verificar Alertas
El plugin procesa eventos de alertas, trading_results, capital_data, macro_data, user_data, y settlement_data, clasificando alertas y enviando notificaciones (logs, correo, Discord).
Consulta las alertas:
docker exec -it corec_v4-alert_db-1 psql -U alert_user -d alert_db -c "SELECT * FROM alerts;"
3. Simular Datos (para pruebas)
Simula eventos para probar:
# Alerta crítica
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('alertas', '{\"data\": \"$(echo '{\"tipo\": \"circuit_breaker_tripped\", \"plugin\": \"trading_execution\"}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# trading_results
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('trading_results', '{\"data\": \"$(echo '{\"profit\": -150.0, \"symbol\": \"BTC/USDT\", \"exchange\": \"binance\", \"user_id\": \"user1\"}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# capital_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('capital_data', '{\"data\": \"$(echo '{\"user_id\": \"user1\", \"action\": \"withdraw\", \"amount\": 600.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# macro_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0, \"vix_price\": 22.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"

# settlement_data
docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('settlement_data', '{\"data\": \"$(echo '{\"roi_percent\": -6.0}' | gzip | base64)\"}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
4. Ejecutar Pruebas
Valida el plugin:
pytest tests/plugins/test_alert_manager.py
5. Revisar Logs
Verifica la salida:
docker logs corec_v4-corec1-1
Funcionalidades
    •	Recopilación: Escucha eventos críticos de alertas y otros canales (trading_results, capital_data, etc.).
    •	Clasificación: Clasifica alertas por severidad (crítica, alta, media, baja) según umbrales configurables.
    •	Notificaciones: Envía alertas a logs, correo, y Discord (futuro), con mensajes personalizados.
    •	Integración: Soporta plugins actuales y futuros mediante el canal alertas.
    •	Eficiencia: Caché en Redis (TTL: 3600 segundos) para evitar duplicados.
    •	Resiliencia: Circuit breakers, reintentos con backoff, fallbacks locales.
Integración con Otros Plugins
    •	Núcleo: Usa CoreCNucleus.publicar_alerta para alertas críticas.
    •	predictor_temporal: Alertas por errores en predicciones.
    •	market_monitor: Alertas por retrasos en precios.
    •	exchange_sync: Alertas por fallos de API.
    •	macro_sync: Alertas por alta volatilidad (VIX).
    •	trading_execution: Alertas por pérdidas/ganancias significativas.
    •	capital_pool: Alertas por movimientos financieros grandes.
    •	user_management: Alertas por actividad de usuarios.
    •	daily_settlement: Alertas por ROI bajo.
    •	Futuros Plugins: Escucha alertas para nuevos eventos.
Extensión
    •	Canales: Añade SMS o Slack en notification_channels.
    •	Umbrales: Personaliza thresholds en alert_manager.yaml.
    •	Plantillas: Define plantillas de mensajes en notify.py.
    •	Análisis: Integra con system_analyzer para análisis avanzado con OpenRouter.
Notas
    •	Plug-and-play: Independiente, usa canales para comunicación.
    •	Base de Datos: Inicializa alert_db antes de usar.
    •	Notificaciones: Configura correo/Discord con credenciales reales en alert_manager.yaml.
    •	Contacto: Consulta al arquitecto principal para dudas sobre CoreC v4.
Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.
---

### Confirmación y Contexto
- **Corrección del Error**: Gracias por señalar que faltaba el README. Ahora está incluido, cubriendo la descripción, configuración, uso, y extensiones del plugin `alert_manager`, asegurando que sea útil para desarrolladores y compatible con CoreC v4.
- **Relevancia de Memorias**: Tu interés en un plugin de alertas generalizado (15/04/2025) inspiró el diseño de `alert_manager`, que usa el canal `alertas` definido en `CoreCNucleus` y es compatible con todos los plugins actuales y futuros, manteniendo la modularidad plug-and-play que enfatizaste.
- **Archivos Previos**: Los archivos del plugin (`plugin.json`, `alert_processor.py`, `db.py`, `notify.py`, `alert_manager.yaml`, `schema.sql`, `test_alert_manager.py`) y las configuraciones (`docker-compose.yml`, `main.py`) ya fueron proporcionados en la respuesta anterior y siguen siendo válidos.

---

### Instrucciones para Implementar
1. **Copiar el README**:
   - Crea el archivo `README.md` en `src/plugins/alert_manager/` y copia el contenido proporcionado.

2. **Verificar Archivos del Plugin**:
   - Asegúrate de que los archivos del plugin (`plugin.json`, `alert_processor.py`, etc.) estén en sus directorios correspondientes (`src/plugins/alert_manager/`, `configs/plugins/alert_manager/`).
   - Confirma que `docker-compose.yml` incluya `alert_db` y que `main.py` tenga la entidad para `alert_manager`.

3. **Configurar Notificaciones**:
   - Edita `alert_manager.yaml` con credenciales reales para correo (SMTP) o Discord (webhook) si deseas habilitar notificaciones externas.
   - Si no usas notificaciones externas, mantén `enabled: false` para `email` y `discord`.

4. **Inicializar `alert_db`**:
   ```bash
   docker cp configs/plugins/alert_manager/schema.sql corec_v4-alert_db-1:/schema.sql
   docker exec corec_v4-alert_db-1 psql -U alert_user -d alert_db -f /schema.sql
    5	Probar el Plugin:
    ◦	Inicia CoreC v4: ./scripts/start.sh
    ◦	
    ◦	Simula eventos (como en la sección “Simular Datos” del README).
    ◦	Verifica alertas en alert_db: docker exec -it corec_v4-alert_db-1 psql -U alert_user -d alert_db -c "SELECT * FROM alerts;"
    ◦	
    ◦	Ejecuta pruebas: pytest tests/plugins/test_alert_manager.py
    ◦	
