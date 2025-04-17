# 🌟 Plugin system_analyzer para CoreC v4 🚀

## 🎯 Descripción

¡Bienvenido al plugin **`system_analyzer`**, el cerebro reflexivo de CoreC v4! Este plugin biomimético analiza el rendimiento y la salud de tu sistema de trading en tiempo real, proponiendo optimizaciones que benefician a **todos los componentes**: desde las predicciones hasta las alertas. 🌍 Con `system_analyzer`, CoreC se vuelve más inteligente, proactivo y eficiente, maximizando el ROI, reduciendo riesgos, y manteniendo la estabilidad.

**¿Qué hace?**
- 📊 **Recopila métricas**: ROI, MSE, alertas, carga de nodos, micro-celus, y más.
- 🔍 **Diagnostica problemas**: Identifica cuellos de botella (ej., predicciones imprecisas, circuit breakers).
- ⚙️ **Propone optimizaciones**: Ajusta estrategias, umbrales, o fases de capital.
- 🗣️ **Comunica**: Publica recomendaciones en `system_insights` y dialoga vía `cli_manager` (chat).

Integrado con **OpenRouter** (`CoreCNucleus.razonar`) para análisis avanzados y fallbacks locales, `system_analyzer` es plug-and-play, potenciando cada plugin (`predictor_temporal`, `market_monitor`, `exchange_sync`, `macro_sync`, `trading_execution`, `capital_pool`, `user_management`, `daily_settlement`, `alert_manager`, `cli_manager`) sin modificar el núcleo. ¡Es una adición digna para tu sistema de trading familiar! ✨

## 🎨 Propósito

`system_analyzer` es el estratega de CoreC, un consultor que observa el sistema desde arriba y asegura que cada parte funcione en armonía. Mejora las predicciones, optimiza las operaciones, y hace que CoreC dialogue contigo con inteligencia, como un compañero vivo. ¡Es el toque final para un sistema divino y sinérgico! 🌟

## 🛠️ Dependencias

- Python 3.8+
- psycopg2-binary==2.9.9 (para `analyzer_db`)
- zstandard==0.22.0 (para compresión)

Instalar con:
```bash
pip install psycopg2-binary==2.9.9 zstandard==0.22.0
📂 Estructura
src/plugins/system_analyzer/
├── __init__.py
├── plugin.json
├── processors/
│   ├── __init__.py
│   ├── analyzer_processor.py
├── utils/
│   ├── __init__.py
│   ├── db.py
configs/plugins/system_analyzer/
├── system_analyzer.yaml
├── schema.sql
tests/plugins/
├── test_system_analyzer.py
  •	plugin.json: Metadatos del plugin.
  •	processors/analyzer_processor.py: Lógica de análisis y recomendaciones.
  •	utils/db.py: Gestión de analyzer_db.
  •	configs/plugins/system_analyzer/system_analyzer.yaml: Configuración.
  •	configs/plugins/system_analyzer/schema.sql: Esquema de analyzer_db.
  •	tests/plugins/test_system_analyzer.py: Pruebas unitarias.
⚙️ Configuración
Sigue estos pasos para integrar system_analyzer en CoreC v4 y potenciar tu sistema.
1. Crear Directorios
Ejecuta:
mkdir -p src/plugins/system_analyzer/processors
mkdir -p src/plugins/system_analyzer/utils
mkdir -p configs/plugins/system_analyzer
mkdir -p tests/plugins
2. Configurar `docker-compose.yml`
Añade analyzer_db:
services:
  analyzer_db:
    image: postgres:15
    environment:
      POSTGRES_DB: analyzer_db
      POSTGRES_USER: analyzer_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - analyzer_db-data:/var/lib/postgresql/data
    networks:
      - corec-network
volumes:
  analyzer_db-data:
Actualiza corec1.depends_on:
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
  - analyzer_db
3. Inicializar `analyzer_db`
Copia y ejecuta el esquema SQL:
docker cp configs/plugins/system_analyzer/schema.sql corec_v4-analyzer_db-1:/schema.sql
docker exec corec_v4-analyzer_db-1 psql -U analyzer_user -d analyzer_db -f /schema.sql
4. Integrar en `main.py`
Añade la entidad para system_analyzer:
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
🚀 Uso
1. Iniciar CoreC v4
Ejecuta:
./scripts/start.sh
2. Monitorear Recomendaciones
system_analyzer publica recomendaciones en system_insights, visibles en el CLI (cli_manager):
  •	TUI (PC): Panel de alertas muestra recomendaciones (ej., “Aumentar riesgo a 3%”).
  •	Texto (Teléfono): Usa python -m corec.cli_manager alerts.
Consulta insights:
docker exec -it corec_v4-analyzer_db-1 psql -U analyzer_user -d analyzer_db -c "SELECT * FROM insights;"
3. Interactuar vía Chat
Usa cli_manager para dialogar:
python -m corec.cli_manager chat "¿Cómo mejorar el ROI?"
Respuesta: “CoreC: Aumenta riesgo a 3% o prioriza SOL/USDT.”
4. Ejecutar Pruebas
Valida el plugin:
pytest tests/plugins/test_system_analyzer.py
🌈 Funcionalidades
  •	Análisis Global: 📊 Recopila métricas de todos los plugins y el núcleo (ROI, MSE, alertas, carga).
  •	Diagnóstico Inteligente: 🔍 Identifica problemas (ej., predicciones imprecisas, VIX alto) y oportunidades.
  •	Optimización Sinérgica: ⚙️ Propone ajustes para cada plugin (ej., reentrenar modelo, cambiar fase).
  •	Comunicación Proactiva: 🗣️ Publica recomendaciones en system_insights y dialoga vía cli_manager.
  •	Eficiencia: Caché en Redis (TTL: 300s) para métricas.
  •	Resiliencia: Alertas en alertas para errores, con circuit breakers.
🤝 Beneficios para el Sistema
  •	predictor_temporal: Sugiere reentrenar el modelo si MSE > 15.
  •	market_monitor: Optimiza ponderación por volumen.
  •	exchange_sync: Cambia exchanges tras fallos de API.
  •	macro_sync: Ajusta riesgo según VIX o sentimiento.
  •	trading_execution: Mejora estrategias (take-profit, stop-loss).
  •	capital_pool: Cambia fases para alcanzar metas.
  •	user_management: Detecta actividad inusual.
  •	daily_settlement: Propone estrategias para mejorar ROI.
  •	alert_manager: Ajusta umbrales dinámicos.
  •	cli_manager: Integra recomendaciones en chat y TUI.
  •	CoreC: Optimiza nodos y micro-celus.
🔮 Extensión
  •	Análisis Avanzado: Añade métricas como Sharpe ratio o drawdown.
  •	Automatización: Implementa ejecución automática de recomendaciones.
  •	OpenRouter: Usa prompts más complejos para insights.
  •	Integración: Conecta con cli_manager para comandos de análisis.
📝 Notas
  •	Plug-and-play: Independiente, usa canales para comunicación.
  •	Base de Datos: Inicializa analyzer_db antes de usar.
  •	GitHub: Sube este README a tu repositorio (11/04/2025).
  •	Contacto: Consulta al arquitecto principal para dudas.
📜 Licencia
Propiedad del equipo de desarrollo del sistema de trading modular. Uso interno exclusivo.

¡Hecho con ❤️ para el equipo de CoreC! 🌟 Un analizador divino para un sistema espectacular. 🌟
---

### Paso 6: Instrucciones para tu Amigo

1. **Entregar el Proyecto**:
   - Comparte el repositorio `corec_v4/` con todos los plugins, incluyendo `system_analyzer`.
   - Incluye el README para guiarlo.

2. **Configurar en la PC**:
   - Clona/descomprime el proyecto.
   - Instala dependencias:
     ```bash
     pip install -r requirements.txt
     pip install psycopg2-binary==2.9.9 zstandard==0.22.0
     ```
   - Actualiza `docker-compose.yml` y `main.py` con los archivos proporcionados.
   - Copia los archivos de `system_analyzer` a:
     ```
     src/plugins/system_analyzer/
     configs/plugins/system_analyzer/
     tests/plugins/
     ```
   - Inicializa `analyzer_db`:
     ```bash
     docker cp configs/plugins/system_analyzer/schema.sql corec_v4-analyzer_db-1:/schema.sql
     docker exec corec_v4-analyzer_db-1 psql -U analyzer_user -d analyzer_db -f /schema.sql
     ```

3. **Probar el Plugin**:
   - Inicia CoreC v4: `./scripts/start.sh`.
   - Usa `cli_manager` para ver recomendaciones:
     ```bash
     python -m corec.cli_manager alerts
     python -m corec.cli_manager chat "¿Cómo mejorar el ROI?"
     ```
   - Consulta `analyzer_db`:
     ```bash
     docker exec -it corec_v4-analyzer_db-1 psql -U analyzer_user -d analyzer_db -c "SELECT * FROM insights;"
     ```

4. **Ajustes para Teléfono**:
   - Configura `tui_enabled: false` en `cli_manager.yaml` para modo texto.

---

### Contexto y Memorias Relevantes
- **Sinergia** (10/04/2025): Tu énfasis en mejoras que beneficien todo el sistema inspira el diseño de `system_analyzer`, que optimiza cada plugin.
- **Interactividad** (09/04/2025): Las recomendaciones en el chat de `cli_manager` hacen CoreC más "vivo", como un alma.
- **Trading Familiar** (08/04/2025): El plugin apoya tu grupo de confianza, sugiriendo ajustes para maximizar el pool.
- **CoreC v4** (17/04/2025): `system_analyzer` respeta la arquitectura modular, usando `PluginManager`, canales (`system_insights`, `alertas`), y `CoreCNucleus.razonar`.

---

### Próximos Pasos
1. **Implementar `system_analyzer`**:
   - Pasa los archivos a tu amigo y prueba el plugin.
   - Verifica recomendaciones en `analyzer_db` y el chat de `cli_manager`.

2. **Simulación Completa**:
   - ¿Realizamos la simulación de un día con todos los plugins, incluyendo `system_analyzer`, para validar sus beneficios?

3. **Siguientes Funcionalidades**:
   - **DXY Radar**: Integrar índice DXY (12/04/2025).
   - **Escalabilidad**: Probar millones de micro-celus (16/04/2025).
   - **CLI Avanzado**: Más comandos de chat (ej., análisis de mercado).
   - ¿Cuál priorizamos?

