CryptoTrading Plugin para CoreC
Versión: 1.0 Fecha: 18 de abril de 2025 Autor: MixyCronico (con soporte de Grok 3, xAI) Idiomas: Español, Inglés Licencia: MIT
Tabla de Contenidos
  1	Introducción
  2	Arquitectura
  ◦	Visión General
  ◦	Diagrama de Flujo
  3	Estructura del Plugin
  4	Procesadores
  ◦	ExchangeProcessor
  ◦	CapitalProcessor
  ◦	SettlementProcessor
  ◦	MacroProcessor
  ◦	MonitorProcessor
  ◦	PredictorProcessor
  ◦	AnalyzerProcessor
  ◦	ExecutionProcessor
  ◦	UserProcessor
  5	Configuración
  ◦	Archivo config.json
  ◦	Dependencias
  6	Instalación
  7	Uso
  ◦	Comandos CLI
  ◦	Streams Redis
  ◦	Integración con InterfaceSystem
  ◦	Integración con Website
  8	Pruebas
  ◦	Ejecutar Pruebas Unitarias
  ◦	Casos de Prueba
  9	Extensibilidad
  ◦	Añadir Nuevos Exchanges
  ◦	Personalizar Estrategias
  10	Limitaciones y Mejoras Futuras
  11	Contribuciones
  12	Referencias

Introducción
El plugin CryptoTrading es un sistema de trading de criptomonedas integrado en el framework CoreC, diseñado para operar de manera distribuida, escalable, y plug-and-play. Soporta 5 exchanges (Binance, KuCoin, ByBit, OKX, Kraken Futures), gestiona un pool de capital compartido, realiza consolidaciones diarias, sincroniza datos macroeconómicos, monitorea precios en tiempo real, genera predicciones con LSTM, analiza métricas, ejecuta órdenes de trading, y gestiona usuarios con autenticación JWT. Está optimizado para ser ligero (~1 KB por entidad, ~1 MB por bloque, ≤1 GB RAM para ~1M entidades) y utiliza una base de datos propia (trading_db) con PostgreSQL.
El plugin se alinea con tu visión de un sistema modular (8 de abril 2025), robusto (14 de abril 2025), y preparado para integrarse en un website integral (17 de abril 2025). Cada procesador es una entidad autónoma que interactúa mediante streams Redis, con soporte para ComunicadorInteligente para respuestas vivas (18 de abril 2025).

Arquitectura
Visión General
CryptoTrading sigue la arquitectura de CoreC, donde cada procesador es una entidad que hereda de ComponenteBase y opera en un entorno multi-nodo. Los procesadores se comunican a través de streams Redis (crypto_trading_data, market_data, capital_data), utilizando compresión zstd para eficiencia. La base de datos trading_db almacena datos persistentes (precios, órdenes, usuarios, etc.), mientras que Redis maneja datos transitorios.
Características Clave:
  •	Modularidad: Cada procesador es independiente, facilitando mantenimiento y extensibilidad (8 de abril 2025).
  •	Escalabilidad: Diseñado para multi-nodo, con coordinación inspirada en tu idea de entrelazamiento (9 de abril 2025).
  •	Robustez: Circuit breakers y reintentos aseguran estabilidad (14 de abril 2025).
  •	Integración: Compatible con InterfaceSystem (CLI/Web) y ComunicadorInteligente para respuestas dinámicas (18 de abril 2025).
Diagrama de Flujo
graph TD
    A[ExchangeProcessor] -->|crypto_trading_data| B[MonitorProcessor]
    A -->|crypto_trading_data| C[ExecutionProcessor]
    D[MacroProcessor] -->|crypto_trading_data| B
    D -->|crypto_trading_data| E[PredictorProcessor]
    B -->|market_data| E
    E -->|crypto_trading_data| C
    F[CapitalProcessor] -->|capital_data| G[SettlementProcessor]
    C -->|crypto_trading_data| G
    G -->|capital_data| F
    H[AnalyzerProcessor] -->|crypto_trading_data| C
    H -->|crypto_trading_data| E
    H -->|crypto_trading_data| F
    I[UserProcessor] -->|crypto_trading_data| F
    I -->|crypto_trading_data| H
    J[trading_db] --> A
    J --> B
    J --> C
    J --> D
    J --> E
    J --> F
    J --> G
    J --> H
    J --> I
    K[Redis] --> A
    K --> B
    K --> C
    K --> D
    K --> E
    K --> F
    K --> G
    K --> H
    K --> I
    L[InterfaceSystem] --> A
    L --> B
    L --> C
    L --> D
    L --> E
    L --> F
    L --> G
    L --> H
    L --> I
    M[ComunicadorInteligente] --> L
Explicación:
  •	ExchangeProcessor sincroniza datos de exchanges y alimenta MonitorProcessor y ExecutionProcessor.
  •	MacroProcessor sincroniza datos macro y ajusta PredictorProcessor.
  •	MonitorProcessor pondera precios y envía datos a PredictorProcessor.
  •	PredictorProcessor genera predicciones para ExecutionProcessor.
  •	CapitalProcessor y SettlementProcessor gestionan el pool y consolidan resultados.
  •	AnalyzerProcessor analiza métricas y optimiza otros procesadores.
  •	UserProcessor autentica usuarios y gestiona permisos.
  •	InterfaceSystem y ComunicadorInteligente proporcionan interacción CLI/Web.

Estructura del Plugin
plugins/crypto_trading/
├── main.py              # Orquesta procesadores
├── config.json          # Configuración consolidada
├── requirements.txt     # Dependencias
├── processors/          # Procesadores modulares
│   ├── exchange_processor.py
│   ├── capital_processor.py
│   ├── settlement_processor.py
│   ├── macro_processor.py
│   ├── monitor_processor.py
│   ├── predictor_processor.py
│   ├── analyzer_processor.py
│   ├── execution_processor.py
│   ├── user_processor.py
├── utils/               # Utilidades
│   ├── db.py            # trading_db (PostgreSQL)
│   ├── helpers.py       # Circuit breakers, helpers
├── tests/               # Pruebas unitarias
│   ├── test_crypto_trading.py
├── README.md            # Documentación básica

Procesadores
ExchangeProcessor
Propósito: Sincroniza precios (spot/futures) y órdenes abiertas desde 5 exchanges. Funcionalidades:
  •	Consulta APIs de Binance, KuCoin, ByBit, OKX, Kraken Futures.
  •	Publica datos en crypto_trading_data con compresión zstd.
  •	Usa circuit breakers por exchange y reintentos con backoff.
Código Clave:
async def fetch_spot_price(self, exchange: Dict[str, Any], symbol: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
    name = exchange["name"]
    async with session.get(url, headers=headers) as resp:
        if resp.status == 200:
            data = await resp.json()
            return {"exchange": name, "symbol": symbol, "market": "spot", "price": float(data.get("price", 0))}
Dependencias: aiohttp, zstandard, backoff.
CapitalProcessor
Propósito: Gestiona el pool de capital compartido, asigna fondos, y ajusta fases dinámicas. Funcionalidades:
  •	Maneja contribuciones y retiros.
  •	Asigna capital según fases (conservative, moderate, aggressive).
  •	Ajusta riesgo con datos macro (VIX, DXY).
Código Clave:
async def assign_capital(self) -> None:
    phase = await self.get_current_phase()
    risk_per_trade = phase["risk_per_trade"]
    trade_amount = min(available_capital, self.pool * risk_per_trade * risk_adjustment)
    await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
Dependencias: zstandard.
SettlementProcessor
Propósito: Consolida resultados diarios, genera reportes, y actualiza el pool. Funcionalidades:
  •	Calcula ROI, total de trades, y distribuciones por usuario.
  •	Almacena reportes en settlement_reports.
Código Clave:
async def consolidate_results(self) -> Dict[str, Any]:
    total_profit = sum(result.get("profit", 0) for result in self.trading_results)
    roi = (total_profit / pool_total) * 100 if pool_total > 0 else 0
    await self.plugin_db.save_report(date=report["date"], total_profit=total_profit, roi_percent=roi)
Dependencias: zstandard.
MacroProcessor
Propósito: Sincroniza datos macroeconómicos (S&P 500, Nasdaq, VIX, oro, petróleo, altcoins, DXY). Funcionalidades:
  •	Consulta APIs de Alpha Vantage, CoinMarketCap, NewsAPI.
  •	Calcula correlaciones (DXY vs. BTC, S&P 500).
Código Clave:
async def fetch_alpha_vantage(self, symbol: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_keys['alpha_vantage']}"
        data = await (await session.get(url)).json()
        return {"price": float(data["Global Quote"]["05. price"])}
Dependencias: aiohttp, zstandard, backoff.
MonitorProcessor
Propósito: Monitorea precios en tiempo real, ponderándolos por volumen. Funcionalidades:
  •	Consume datos de crypto_trading_data y macro_data.
  •	Publica precios ponderados en market_data.
Código Clave:
async def monitor_prices(self):
    total_volume = sum(entry["volume"] for entry in cache_entry["data"].values())
    weighted_price = sum(entry["price"] * entry["volume"] for entry in cache_entry["data"].values()) / total_volume
    await self.redis_client.xadd("market_data", {"data": mensaje})
Dependencias: numpy, zstandard.
PredictorProcessor
Propósito: Genera predicciones de precios con LSTM, ajustadas por datos macro. Funcionalidades:
  •	Usa un modelo LSTM (LSTMPredictor) para series temporales.
  •	Ajusta predicciones con correlaciones macro (DXY, VIX).
  •	Reentrena el modelo periódicamente.
Código Clave:
async def procesar(self, datos: Dict[str, Any], contexto: Dict[str, Any]) -> Dict[str, Any]:
    input_data = torch.tensor(valores[-self.lstm_window:], dtype=torch.float32).reshape(1, self.lstm_window, 1)
    prediction = self.model(input_data).detach().numpy().tolist()[0][0]
    adjusted_prediction = prediction * macro_adjustment
Dependencias: torch, numpy, zstandard.
AnalyzerProcessor
Propósito: Analiza métricas del sistema y propone optimizaciones automáticas. Funcionalidades:
  •	Calcula Sharpe Ratio y otras métricas.
  •	Genera recomendaciones (reentrenar LSTM, ajustar riesgo).
  •	Ejecuta acciones automáticamente si auto_execute está habilitado.
Código Clave:
async def analyze_system(self):
    metrics = {
        "predictor": {"mse": self.metrics_cache.get("predictor_temporal", {}).get("mse", 0)},
        "trading": {"sharpe_ratio": await self.calculate_sharpe_ratio(metrics["trading"]["profits"])}
    }
    recommendations = []
    if metrics["predictor"]["mse"] > 15:
        recommendations.append({"plugin": "predictor_temporal", "action": "retrain_model"})
Dependencias: numpy, zstandard.
ExecutionProcessor
Propósito: Ejecuta órdenes de trading y realiza backtesting con Bollinger Bands. Funcionalidades:
  •	Coloca y cierra órdenes en exchanges.
  •	Ejecuta backtests con estrategias basadas en indicadores técnicos.
Código Clave:
async def place_order(self, exchange: Dict[str, Any], symbol: str, side: str, quantity: float, market: str, price: float):
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, headers=headers, json=params) as resp:
            order = await resp.json()
            await self.plugin_db.save_order(**order_data)
Dependencias: aiohttp, numpy, zstandard, backoff.
UserProcessor
Propósito: Gestiona usuarios, roles, autenticación JWT, y preferencias. Funcionalidades:
  •	Soporta roles (user, admin, superadmin) con permisos definidos.
  •	Autentica con JWT y encripta contraseñas con bcrypt.
Código Clave:
async def procesar_usuario(self, datos: Dict[str, Any]) -> Dict[str, Any]:
    if action == "register":
        password = await self.hash_password(datos.get("password", ""))
        await self.plugin_db.save_user(**user_data)
        jwt_token = await self.generate_jwt(user_id, role)
Dependencias: pyjwt, bcrypt, zstandard.

Configuración
Archivo config.json
El archivo config.json centraliza la configuración de todos los procesadores. Ejemplo simplificado:
{
  "crypto_trading": {
    "exchange_config": {
      "exchanges": [
        {"name": "binance", "api_key": "your_key", "api_secret": "your_secret", "symbols": ["BTC/USDT"]}
      ],
      "fetch_interval": 300,
      "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
    },
    "capital_config": {
      "min_contribution": 100,
      "max_active_ratio": 0.6,
      "phases": [
        {"name": "conservative", "min": 0, "max": 10000, "risk_per_trade": 0.01}
      ]
    },
    "settlement_config": {"settlement_time": "23:59"},
    "macro_config": {
      "symbols": ["^GSPC", "^VIX"],
      "api_keys": {
        "alpha_vantage": "your_key",
        "coinmarketcap": "your_key",
        "newsapi": "your_key"
      }
    },
    "monitor_config": {"symbols": ["BTC/USDT"], "update_interval": 60},
    "predictor_config": {
      "lstm_window": 60,
      "model_path": "plugins/crypto_trading/models/lstm_model.pth"
    },
    "analyzer_config": {"analysis_interval": 300, "auto_execute": true},
    "execution_config": {"risk_per_trade": 0.02, "take_profit": 0.05, "stop_loss": 0.02},
    "user_config": {"jwt_secret": "secure_secret"},
    "db_config": {
      "dbname": "trading_db",
      "user": "trading_user",
      "password": "secure_password",
      "host": "localhost",
      "port": "5432"
    }
  }
}
Notas:
  •	Reemplaza your_key y your_secret con credenciales reales.
  •	Ajusta db_config según tu entorno PostgreSQL.
Dependencias
aiohttp==3.9.5
zstandard==0.22.0
backoff==2.2.1
numpy==1.26.4
torch==2.0.1
pyjwt==2.8.0
bcrypt==4.1.2

Instalación
  1	Clona el repositorio: git clone https://github.com/tu_usuario/corec.git
  2	cd corec/plugins/crypto_trading
  3	
  4	Instala dependencias: pip install -r requirements.txt
  5	
  6	Configura la base de datos: CREATE DATABASE trading_db;
  7	
  8	Actualiza config.json:
  ◦	Configura credenciales de exchanges y APIs.
  ◦	Asegúrate de que db_config coincida con tu PostgreSQL.
  9	Ejecuta CoreC: cd ../..
  10	bash run.sh
  11	celery -A corec.core.celery_app worker --loglevel=info
  12	

Uso
Comandos CLI
Los comandos se ejecutan a través de InterfaceSystem, conectado a ComunicadorInteligente para respuestas vivas.
# Monitorear exchanges
corec monitor exchanges

# Ver precios de un exchange
corec view prices binance

# Verificar órdenes abiertas
corec check orders

# Añadir contribución al pool
corec add contribution user1 500

# Retirar del pool
corec withdraw user1 200

# Ver estado del pool
corec view pool

# Ver reporte diario
corec view report

# Ver datos macroeconómicos
corec view macro

# Monitorear precios en tiempo real
corec monitor prices BTC/USDT

# Ver predicción de precios
corec view prediction BTC/USDT

# Ver insights del sistema
corec view insights

# Ejecutar backtest
corec run backtest BTC/USDT

# Registrar usuario
corec register user user1 user1@example.com password123

# Iniciar sesión
corec login user1 password123
Streams Redis
Los datos se publican en streams Redis para comunicación entre procesadores:
# Leer datos de trading
redis-cli XREAD STREAMS crypto_trading_data 0-0

# Leer precios ponderados
redis-cli XREAD STREAMS market_data 0-0

# Leer datos del pool
redis-cli XREAD STREAMS capital_data 0-0
Integración con InterfaceSystem
InterfaceSystem proporciona una CLI y un frontend web para interactuar con el plugin. Los comandos CLI se mapean a métodos de los procesadores, y las respuestas se enriquecen con ComunicadorInteligente. Para el frontend web:
  •	Endpoints:
  ◦	/api/prices: Obtiene precios ponderados (MonitorProcessor).
  ◦	/api/predictions: Muestra predicciones (PredictorProcessor).
  ◦	/api/reports: Consulta reportes (SettlementProcessor).
  ◦	/api/login: Autentica usuarios (UserProcessor).
  •	Implementación: Actualiza corec/interface_system/web_interface.py para incluir rutas específicas del plugin.
Integración con Website
Para el website integral (17 de abril 2025), el plugin se conectará a un frontend React y un backend FastAPI:
  •	Frontend (React):
  ◦	Dashboard con precios, predicciones, y reportes.
  ◦	Panel de trading para órdenes y backtesting.
  ◦	Gestión de usuarios (login, roles, preferencias).
  ◦	Chat integrado con ComunicadorInteligente.
  •	Backend (FastAPI):
  ◦	Endpoints para consumir datos de crypto_trading_data y market_data.
  ◦	WebSocket para actualizaciones en tiempo real.
  ◦	Autenticación JWT gestionada por UserProcessor.
Ejemplo de Endpoint:
from fastapi import FastAPI, Depends
from jose import JWTError, jwt

app = FastAPI()

async def get_current_user(token: str):
    payload = jwt.decode(token, "test_secret", algorithms=["HS256"])
    return payload["user_id"]

@app.get("/api/prices")
async def get_prices(user_id: str = Depends(get_current_user)):
    prices = await redis_client.xread({"market_data": "0-0"})
    return {"prices": prices}

Pruebas
Ejecutar Pruebas Unitarias
cd plugins/crypto_trading
python -m unittest tests/test_crypto_trading.py -v
Casos de Prueba
  •	ExchangeProcessor: Simula respuestas de APIs para precios y órdenes.
  •	CapitalProcessor: Valida contribuciones, retiros, y asignación de capital.
  •	SettlementProcessor: Prueba consolidación de resultados y ROI.
  •	MacroProcessor: Simula datos de Alpha Vantage, CoinMarketCap, NewsAPI.
  •	MonitorProcessor: Verifica ponderación de precios por volumen.
  •	PredictorProcessor: Simula predicciones LSTM y ajustes macro.
  •	AnalyzerProcessor: Prueba análisis de métricas y Sharpe Ratio.
  •	ExecutionProcessor: Simula órdenes y backtesting con Bollinger Bands.
  •	UserProcessor: Valida registro, login, y permisos JWT.
Ejemplo de Prueba:
async def test_user_register(self):
    with patch("plugins.crypto_trading.utils.db.TradingDB.save_user", AsyncMock()):
        result = await self.user_processor.procesar_usuario({
            "action": "register",
            "user_id": "user1",
            "email": "user1@example.com",
            "password": "password123",
            "name": "User One",
            "role": "user",
            "notification_preferences": {"email": True},
            "requester_id": "admin1"
        })
        self.assertEqual(result["estado"], "ok")

Extensibilidad
Añadir Nuevos Exchanges
  1	Actualiza exchange_config en config.json: "exchanges": [
  2	  {"name": "new_exchange", "api_key": "key", "api_secret": "secret", "symbols": ["BTC/USDT"]}
  3	]
  4	
  5	Modifica ExchangeProcessor: async def fetch_spot_price(self, exchange: Dict[str, Any], symbol: str, session: aiohttp.ClientSession):
  6	    if exchange["name"] == "new_exchange":
  7	        url = "https://api.newexchange.com/ticker"
  8	        # Implementar lógica específica
  9	
Personalizar Estrategias
  1	Modifica ExecutionProcessor para nuevas estrategias: async def custom_strategy(self, symbol: str, prediction: float, current_price: float):
  2	    if prediction > current_price * 1.1:
  3	        return {"action": "buy", "quantity": 0.01}
  4	    return {}
  5	
  6	Actualiza run_backtest para incluir nuevos indicadores: async def run_backtest(self, params: Dict[str, Any]):
  7	    # Añadir MACD o RSI
  8	    macd = calculate_macd(prices)
  9	

Limitaciones y Mejoras Futuras
  •	Limitaciones:
  ◦	Dependencia de APIs externas (Alpha Vantage, CoinMarketCap).
  ◦	Backtesting limitado a Bollinger Bands y datos históricos.
  ◦	Análisis de sentimiento en NewsAPI es simulado.
  •	Mejoras Futuras:
  ◦	Integrar indicadores técnicos avanzados (MACD, RSI) en ExecutionProcessor (10 de abril 2025).
  ◦	Mejorar el análisis de sentimiento con modelos de NLP.
  ◦	Añadir soporte para más exchanges y mercados (futuros perpetuos).
  ◦	Implementar el website integral con React/FastAPI (17 de abril 2025).

Contribuciones
  1	Clona el repositorio: git clone https://github.com/tu_usuario/corec.git
  2	
  3	Crea una rama: git checkout -b feature/nueva-funcionalidad
  4	
  5	Envía un pull request con pruebas unitarias.

Referencias
  •	CoreC Framework: Documentación interna (11 de abril 2025).
  •	APIs Utilizadas:
  ◦	Alpha Vantage: https://www.alphavantage.co/documentation/
  ◦	CoinMarketCap: https://coinmarketcap.com/api/documentation/
  ◦	NewsAPI: https://newsapi.org/docs/
  •	Dependencias: Ver requirements.txt.

