# Plugin predictor_temporal para CoreC v4

## Descripción
Plugin biomimético para CoreC v4 que genera predicciones de series temporales (ej., precios de criptomonedas) usando una red neuronal LSTM, ajustadas por datos macroeconómicos desde el canal `macro_data`. Almacena predicciones y métricas (MSE, MAE) en `predictor_db`.

## Propósito
Proveer predicciones precisas para el sistema de trading modular, consumidas por otros plugins (ej., `trading_execution`).

## Dependencias
- Python 3.8+
- torch==2.3.1
- psycopg2-binary==2.9.9
- numpy==1.26.4
- zstandard==0.22.0

Instalar con:
```bash
pip install torch==2.3.1 psycopg2-binary==2.9.9 numpy==1.26.4 zstandard==0.22.0
Estructura
  •	plugin.json: Metadatos del plugin.
  •	processors/predictor_processor.py: Lógica de predicciones con LSTM.
  •	utils/db.py: Gestión de predictor_db.
  •	configs/plugins/predictor_temporal/predictor_temporal.yaml: Configuración.
  •	configs/plugins/predictor_temporal/schema.sql: Esquema de la base de datos.
  •	tests/plugins/test_predictor_temporal.py: Pruebas unitarias.
Configuración
  1	Crear directorios: mkdir -p src/plugins/predictor_temporal/processors
  2	mkdir -p src/plugins/predictor_temporal/utils
  3	mkdir -p configs/plugins/predictor_temporal
  4	mkdir -p tests/plugins
  5	
  6	Configurar docker-compose.yml: Añadir predictor_db: services:
  7	  predictor_db:
  8	    image: postgres:15
  9	    environment:
  10	      POSTGRES_DB: predictor_db
  11	      POSTGRES_USER: predictor_user
  12	      POSTGRES_PASSWORD: secure_password
  13	    volumes:
  14	      - predictor_db-data:/var/lib/postgresql/data
  15	    networks:
  16	      - corec-network
  17	volumes:
  18	  predictor_db-data:
  19	
  20	Inicializar predictor_db: docker cp configs/plugins/predictor_temporal/schema.sql corec_v4-predictor_db-1:/schema.sql
  21	docker exec corec_v4-predictor_db-1 psql -U predictor_user -d predictor_db -f /schema.sql
  22	
  23	Integrar en main.py: await nucleus.registrar_celu_entidad(
  24	    CeluEntidadCoreC(
  25	        f"nano_predictor_{instance_id}",
  26	        nucleus.get_procesador("predictor_temporal"),
  27	        "predictor_temporal",
  28	        5.0,
  29	        nucleus.db_config,
  30	        instance_id=instance_id
  31	    )
  32	)
  33	
Uso
  1	Iniciar CoreC v4: ./scripts/start.sh
  2	
  3	Simular datos: docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('predictor_temporal', '{\"valores\": [35000, 35010, 35020, 35030, 35040, 35050, 35060, 35070, 35080, 35090, 35100, 35110, 35120, 35130, 35140, 35150, 35160, 35170, 35180, 35190, 35200, 35210, 35220, 35230, 35240, 35250, 35260, 35270, 35280, 35290, 35300, 35310, 35320, 35330, 35340, 35350, 35360, 35370, 35380, 35390, 35400, 35410, 35420, 35430, 35440, 35450, 35460, 35470, 35480, 35490, 35500, 35510, 35520, 35530, 35540, 35550, 35560, 35570, 35580, 35590], \"symbol\": \"BTC/USDT\", \"actual_value\": 35600}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  4	
  5	Simular datos macro (temporal, hasta implementar macro_sync): docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('macro_data', '{\"sp500_price\": 4500.0}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  6	
  7	Verificar resultados: docker exec -it corec_v4-predictor_db-1 psql -U predictor_user -d predictor_db -c "SELECT * FROM predictions;"
  8	docker exec -it corec_v4-predictor_db-1 psql -U predictor_user -d predictor_db -c "SELECT * FROM metrics;"
  9	
  10	Ejecutar pruebas: pytest tests/plugins/test_predictor_temporal.py
  11	
Extensión
  •	Añadir macro datos: Implementar macro_sync para publicar S&P 500, DXY, etc., en macro_data.
  •	Integrar con trading: Conectar con trading_execution para usar predicciones en trades.
  •	Optimización: Ajustar hiperparámetros del LSTM o añadir métricas avanzadas (Sharpe Ratio).
Notas
  •	El plugin es plug-and-play, no depende de otros plugins, solo de canales (predictor_temporal, macro_data).
  •	Usa circuit breaker y fallbacks para resiliencia.
  •	Contactar al arquitecto principal para dudas sobre CoreC v4 o integración con otros plugins.
---

