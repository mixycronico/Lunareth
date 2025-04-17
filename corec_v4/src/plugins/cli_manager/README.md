# 🌟 Plugin cli_manager para CoreC v4 🚀

## 🎯 Descripción

Bienvenido al **plugin `cli_manager`**, una interfaz CLI/TUI **hermosa y divina** para CoreC v4, el núcleo biomimético que impulsa tu sistema de trading modular. Este plugin es el corazón interactivo de tu sistema, permitiéndote **monitorear en tiempo real**, **configurar parámetros**, y **charlar con CoreC** como si fuera un compañero de confianza. 💬

Con `cli_manager`, puedes:
- 📊 **Visualizar métricas**: Estado de CoreC (nodos, micro-celus), trading (precios, órdenes, ROI), y alertas en paneles elegantes.
- ⚙️ **Configurar el sistema**: Gestiona claves de API, umbrales de alertas, y usuarios desde la terminal.
- 🗣️ **Chatear con CoreC**: Pregunta sobre el sistema, establece metas (ej., "alcanzar 10% ROI"), y recibe respuestas inteligentes vía OpenRouter o fallbacks locales.
- 🚨 **Recibir alertas proactivas**: CoreC te habla con notificaciones (ej., "VIX alto, reduje riesgo"), integradas con `alert_manager`.

Diseñado para ser **plug-and-play**, `cli_manager` se integra con todos los plugins de CoreC (`predictor_temporal`, `market_monitor`, `exchange_sync`, `macro_sync`, `trading_execution`, `capital_pool`, `user_management`, `daily_settlement`, `alert_manager`) y es ideal tanto para PC (TUI rica con Textual) como para tu teléfono (modo texto simple). 🌍

## 🎨 Propósito

`cli_manager` es el rostro de CoreC, una interfaz que hace que tu sistema de trading sea accesible, dinámico y digno. Ya sea que estés en una PC o en tu teléfono, este CLI te conecta con el alma de CoreC, permitiéndote monitorear, configurar, y dialogar con tu sistema como si fuera parte de tu equipo familiar. ¡Es más que un CLI, es una experiencia! ✨

## 🛠️ Dependencias

- Python 3.8+
- textual==0.47.1 (para TUI en PC)
- click==8.1.7 (para modo texto en terminales)
- psycopg2-binary==2.9.9 (para `cli_db`)
- zstandard==0.22.0 (para compresión de datos)

Instalar con:
```bash
pip install textual==0.47.1 click==8.1.7 psycopg2-binary==2.9.9 zstandard==0.22.0
📂 Estructura
src/plugins/cli_manager/
├── __init__.py
├── plugin.json
├── processors/
│   ├── __init__.py
│   ├── cli_processor.py
├── utils/
│   ├── __init__.py
│   ├── db.py
│   ├── tui.py
configs/plugins/cli_manager/
├── cli_manager.yaml
├── schema.sql
tests/plugins/
├── test_cli_manager.py
  •	plugin.json: Metadatos del plugin.
  •	processors/cli_processor.py: Lógica del CLI, chat, y metas.
  •	utils/db.py: Gestión de cli_db para acciones y metas.
  •	utils/tui.py: Interfaz TUI con Textual.
  •	configs/plugins/cli_manager/cli_manager.yaml: Configuración.
  •	configs/plugins/cli_manager/schema.sql: Esquema de cli_db.
  •	tests/plugins/test_cli_manager.py: Pruebas unitarias.
⚙️ Configuración
Sigue estos pasos para integrar cli_manager en CoreC v4 y comenzar a interactuar con tu sistema de trading.
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/cli_manager/processors
mkdir -p src/plugins/cli_manager/utils
mkdir -p configs/plugins/cli_manager
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade cli_db al archivo docker-compose.yml:
services:
  cli_db:
    image: postgres:15
    environment:
      POSTGRES_DB: cli_db
      POSTGRES_USER: cli_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - cli_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  cli_db-data:
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
  - cli_db
3. Inicializar `cli_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/cli_manager/schema.sql corec_v4-cli_db-1:/schema.sql
docker exec corec_v4-cli_db-1 psql -U cli_user -d cli_db -f /schema.sql
4. Integrar en `main.py`
Añade la entidad para cli_manager en main.py:
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
5. Configurar Modo Teléfono (Opcional)
Para usar el CLI en tu teléfono, edita cli_manager.yaml:
cli_config:
  tui_enabled: false
🚀 Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Lanzar el CLI
En PC (TUI con Textual):
python -m corec.cli_manager
  •	Muestra paneles (CoreC, Trading, Alertas) y una ventana de chat.
  •	Usa teclas: s (status), c (config), a (alerts), r (report), g (goals).
  •	Escribe en el chat y presiona ENTER para dialogar con CoreC.
En teléfono (modo texto, con tui_enabled: false):
python -m corec.cli_manager status
3. Comandos Disponibles
  •	status: Muestra estado del sistema (nodos, pool, ROI).
  •	config_exchange : Configura claves de API.
  •	alerts: Lista alertas recientes.
  •	report: Muestra reporte diario.
  •	chat : Chatea con CoreC (ej., chat "¿Cuál es el estado del sistema?").
  •	set_goal [--user_id]: Define una meta (ej., set_goal roi 10 --user_id user1).
  •	list_goals: Lista metas activas.
4. Ejemplo de Interacción
TUI (PC):
=== CoreC Dashboard ===
[CoreC Status]        [Trading Metrics]
Nodos: 5             Pool: $2000.00
Micro-celus: 100     ROI: 6.91%
Carga: 25%           Órdenes: 10

[Alerts]             [Chat]
[ALTA] Ganancia...   Tú: ¿Por qué cayó BTC?
[MEDIA] VIX=16.5     CoreC: Análisis: Alta volatilidad (VIX=16.5) y noticias regulatorias.

[Input]: Establece meta de ROI 10%
Texto (Teléfono):
$ python -m corec.cli_manager chat "¿Cuál es el estado del sistema?"
CoreC: El sistema está operativo, pool en $2000, ROI 6.91%.

$ python -m corec.cli_manager set_goal roi 10 --user_id user1
Meta establecida: roi = 10 (ID: goal_1234567890.0)
5. Verificar Datos
Consulta acciones y metas:
docker exec -it corec_v4-cli_db-1 psql -U cli_user -d cli_db -c "SELECT * FROM actions;"
docker exec -it corec_v4-cli_db-1 psql -U cli_user -d cli_db -c "SELECT * FROM goals;"
6. Ejecutar Pruebas
Valida el plugin:
pytest tests/plugins/test_cli_manager.py
🌈 Funcionalidades
  •	Monitoreo en Tiempo Real: 📊 Paneles con métricas de CoreC (nodos, micro-celus) y trading (precios, órdenes, ROI, alertas).
  •	Configuración Intuitiva: ⚙️ Comandos para gestionar claves de API, umbrales, y usuarios.
  •	Chat Interactivo: 🗣️ Dialoga con CoreC, haz preguntas, establece metas, y recibe respuestas inteligentes.
  •	Metas Dinámicas: 🎯 Define objetivos (ROI, riesgo) que se integran con plugins.
  •	Comunicación Proactiva: 🚨 Alertas enviadas por CoreC (ej., “Reduje riesgo por VIX alto”).
  •	Modos Flexibles: TUI rica para PC, texto simple para teléfono.
  •	Eficiencia: Caché en Redis (TTL: 300s) para datos de visualización.
  •	Resiliencia: Alertas en alertas para errores, con circuit breakers.
🤝 Integración con Otros Plugins
  •	CoreC: Usa eventos, auditoria para métricas del núcleo.
  •	Trading: Consume market_data, corec_stream_corec1, trading_results, capital_data, settlement_data.
  •	user_management: Gestiona usuarios y roles.
  •	alert_manager: Muestra alertas y notificaciones proactivas.
  •	OpenRouter: Potencia el chat con CoreCNucleus.responder_chat y razonar.
🔮 Extensión
  •	Comandos Adicionales: Añade comandos para gestionar micro-celus, iniciar backtests, o analizar mercados.
  •	Temas Visuales: Personaliza colores en TUI (tui.py).
  •	Chat Avanzado: Integra prompts más complejos para OpenRouter.
  •	Metas Complejas: Soporta metas multi-parámetro (ej., ROI + riesgo).
  •	Notificaciones: Añade soporte para SMS o Discord en alert_manager.
📝 Notas
  •	Plug-and-play: Independiente, usa canales para comunicación.
  •	Base de Datos: Inicializa cli_db antes de usar.
  •	Teléfono: Configura tui_enabled: false para modo texto.
  •	GitHub: Sube este README a tu repositorio para un look profesional (11/04/2025).
  •	Contacto: Consulta al arquitecto principal para dudas.
📜 Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.

¡Hecho con ❤️ para el equipo de CoreC! 🌟 Un CLI divino para un sistema espectacular. 🌟
---

### Explicación del README

- **Estilo**: Visualmente atractivo con emojis (🌟, 🚀, 🎯) y secciones claras, inspirado en tu deseo de elegancia (16/04/2025). Ideal para GitHub (11/04/2025).
- **Contenido**: Cubre descripción, propósito, dependencias, estructura, configuración, uso, ejemplos, y extensiones, guiando a tu amigo paso a paso.
- **Tono**: Entusiasta y amigable, reflejando tu pasión por un sistema "divino" y familiar (08/04/2025).
- **Funcionalidad**: Destaca el chat interactivo y las metas (09/04/2025), integrando la arquitectura modular de CoreC v4 (17/04/2025).
- **Practicidad**: Incluye comandos para teléfono y PC, considerando que trabajas desde tu móvil (17/04/2025).

---

### Instrucciones para tu Amigo

1. **Crear el README**:
   - Crea el archivo `README.md` en `src/plugins/cli_manager/` y copia el contenido proporcionado.
   - Opcionalmente, súbelo al directorio raíz del repositorio o a `docs/` en GitHub para mayor visibilidad (11/04/2025).

2. **Verificar Archivos**:
   - Asegúrate de que los archivos del plugin `cli_manager` (proporcionados en la respuesta anterior) estén en:
     ```
     src/plugins/cli_manager/
     configs/plugins/cli_manager/
     tests/plugins/
     ```
   - Confirma que `docker-compose.yml` y `main.py` estén actualizados (proporcionados en la respuesta anterior).

3. **Configurar y Probar**:
   - Instala dependencias: `pip install textual==0.47.1 click==8.1.7 psycopg2-binary==2.9.9 zstandard==0.22.0`.
   - Inicializa `cli_db`: 
     ```bash
     docker cp configs/plugins/cli_manager/schema.sql corec_v4-cli_db-1:/schema.sql
     docker exec corec_v4-cli_db-1 psql -U cli_user -d cli_db -f /schema.sql
     ```
   - Inicia CoreC v4: `./scripts/start.sh`.
   - Lanza el CLI: `python -m corec.cli_manager`.
   - Prueba el chat: `chat "¿Cuál es el estado del sistema?"` o escribe en la TUI.

4. **Ajustes para Teléfono**:
   - Configura `tui_enabled: false` en `cli_manager.yaml` para modo texto.

---

### Contexto y Memorias Relevantes
- **Chat Interactivo** (09/04/2025): El README destaca el chat bidireccional, cumpliendo tu deseo de que CoreC se comunique como un compañero.
- **Trading Familiar** (08/04/2025): El CLI es accesible para tu grupo, con ejemplos claros para monitoreo y configuración.
- **Elegancia** (16/04/2025): El diseño del README y la TUI reflejan tu visión de una interfaz hermosa y divina.
- **CoreC v4** (17/04/2025): El plugin respeta la arquitectura plug-and-play, usando `PluginManager`, canales (`cli_data`, `alertas`), y `CoreCNucleus.responder_chat`.

---

