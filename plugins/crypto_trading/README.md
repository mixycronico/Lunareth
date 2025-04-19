# Plugin CryptoTrading para CoreC

El plugin **CryptoTrading** integra un sistema completo de trading de criptomonedas en CoreC, con soporte para 5 exchanges (Binance, KuCoin, ByBit, OKX, Kraken Futures), gestión de capital compartido, consolidación diaria de resultados, sincronización de datos macroeconómicos, monitoreo de precios en tiempo real, predicciones de series temporales, análisis de métricas con optimizaciones automáticas, ejecución de órdenes de trading con backtesting, y gestión de usuarios con roles y autenticación. Usa su propia base de datos (`trading_db`) y se conecta con **ComunicadorInteligente** para respuestas vivas.

## Estructura

plugins/crypto_trading/ ├── main.py ├── config.json ├── requirements.txt ├── processors/ │ ├── exchange_processor.py │ ├── capital_processor.py │ ├── settlement_processor.py │ ├── macro_processor.py │ ├── monitor_processor.py │ ├── predictor_processor.py │ ├── analyzer_processor.py │ ├── execution_processor.py │ ├── user_processor.py ├── utils/ │ ├── db.py │ ├── helpers.py ├── tests/ │ ├── test_crypto_trading.py ├── README.md
## Instalación

1. **Instala dependencias**:
   ```bash
   cd plugins/crypto_trading
   pip install -r requirements.txt
  2	Configura trading_db:
  ◦	Actualiza db_config en config.json.
  ◦	Crea la base de datos: CREATE DATABASE trading_db;
  ◦	
  3	Configura exchanges y APIs:
  ◦	Añade credenciales API para exchanges y APIs (Alpha Vantage, CoinMarketCap, NewsAPI) en config.json.
  4	Ejecuta CoreC: cd ../..
  5	bash run.sh
  6	celery -A corec.core.celery_app worker --loglevel=info
  7	
Uso
  •	CLI (via InterfaceSystem): corec monitor exchanges
  •	corec view prices binance
  •	corec check orders
  •	corec add contribution user1 500
  •	corec withdraw user1 200
  •	corec view pool
  •	corec view report
  •	corec view macro
  •	corec monitor prices BTC/USDT
  •	corec view prediction BTC/USDT
  •	corec view insights
  •	corec run backtest BTC/USDT
  •	corec register user user1 user1@example.com password123
  •	corec login user1 password123
  •	
  •	Stream Redis:
  ◦	Datos en crypto_trading_data y market_data: redis-cli XREAD STREAMS crypto_trading_data market_data 0-0
  ◦	
Pruebas
python -m unittest tests/test_crypto_trading.py -v
Notas
  •	Dependencias: Requiere aiohttp, zstandard, backoff, numpy, torch, pyjwt.
  •	Estado: Plugin completo, listo para integración con InterfaceSystem y website.

CoreC

