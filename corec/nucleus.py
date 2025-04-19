from corec.core import logging, json, os, importlib, zstd, psycopg2, aioredis, celery_app, ComponenteBase, CANALES_CRITICOS
from corec.entities import CeluEntidadCoreC, MicroCeluEntidadCoreC
from pathlib import Path
from typing import Dict, Any

class CoreCNucleus(ComponenteBase):
    def __init__(self, config_path: str, instance_id: str):
        self.logger = logging.getLogger(f"CoreCNucleus-{instance_id}")
        self.instance_id = instance_id
        self.config = self._cargar_config(config_path)
        self.modulos = {}
        self.plugins = {}
        self.db_config = self.config.get("db_config", {})
        self.redis_config = self.config.get("redis_config", {
            "host": "redis",
            "port": 6379,
            "username": "corec_user",
            "password": "secure_password"
        })
        self.redis_client = None
        self.canales_criticos = CANALES_CRITICOS
        self.rol = self.config.get("rol", "generica")

    def _cargar_config(self, config_path: str) -> Dict[str, Any]:
        config_key = f"config_{self.instance_id}"
        try:
            if self.redis_client:
                cached_config = self.redis_client.get(config_key)
                if cached_config:
                    self.logger.info("Configuración cargada desde caché")
                    return json.loads(cached_config)
            with open(config_path, "r") as f:
                config = json.load(f)
            if self.redis_client:
                self.redis_client.set(config_key, json.dumps(config), ex=3600)
            return config
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
            return {"db_config": {}, "redis_config": {"host": "redis", "port": 6379, "username": "corec_user", "password": "secure_password"}}

    async def _inicializar_redis(self):
        try:
            redis_url = f"redis://{self.redis_config['username']}:{self.redis_config['password']}@{self.redis_config['host']}:{self.redis_config['port']}"
            self.redis_client = await aioredis.from_url(redis_url, decode_responses=True)
            self.logger.info("Cliente Redis inicializado")
        except Exception as e:
            self.logger.error(f"Error inicializando Redis: {e}")
            raise

    async def _inicializar_db(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bloques (
                    id TEXT PRIMARY KEY,
                    canal TEXT,
                    num_entidades INTEGER,
                    fitness REAL,
                    timestamp REAL,
                    instance_id TEXT
                ) PARTITION BY RANGE (timestamp)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bloques_2025_04 PARTITION OF bloques
                FOR VALUES FROM (UNIX_TIMESTAMP('2025-04-01')) TO (UNIX_TIMESTAMP('2025-05-01'))
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_bloques_canal_timestamp ON bloques (canal, timestamp DESC)")
            conn.commit()
            cur.close()
            conn.close()
            self.logger.info("Base de datos PostgreSQL inicializada")
        except Exception as e:
            self.logger.error(f"Error inicializando PostgreSQL: {e}")
            raise

    def cargar_modulos(self, directorio: str = "modules"):
        directorio_path = Path(__file__).parent / directorio
        if not directorio_path.exists():
            self.logger.warning(f"Directorio {directorio} no existe")
            return
        for archivo in directorio_path.glob("*.py"):
            if archivo.name.startswith("__"):
                continue
            modulo_nombre = archivo.stem
            try:
                modulo = importlib.import_module(f"corec.{directorio}.{modulo_nombre}")
                instancia = modulo.Modulo()
                self.modulos[modulo_nombre] = instancia
                self.logger.info(f"Módulo {modulo_nombre} cargado")
            except Exception as e:
                self.logger.error(f"Error cargando módulo {modulo_nombre}: {e}")

    def cargar_plugins(self, directorio: str = "plugins"):
        directorio_path = Path(__file__).parent.parent / directorio
        if not directorio_path.exists():
            self.logger.warning(f"Directorio {directorio} no existe")
            return
        for plugin_dir in directorio_path.glob("*/"):
            if not plugin_dir.is_dir():
                continue
            plugin_nombre = plugin_dir.name
            main_path = plugin_dir / "main.py"
            if not main_path.exists():
                self.logger.warning(f"Plugin {plugin_nombre} no tiene main.py")
                continue
            try:
                config_path = plugin_dir / "config.json"
                config = {}
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                if config.get("enabled", True):
                    plugin = importlib.import_module(f"plugins.{plugin_nombre}.main")
                    plugin.inicializar(self, config)
                    self.logger.info(f"Plugin {plugin_nombre} cargado")
            except ImportError as e:
                self.logger.error(f"Error de dependencias en plugin {plugin_nombre}: {e}")
            except Exception as e:
                self.logger.error(f"Error cargando plugin {plugin_nombre}: {e}")

    def registrar_plugin(self, nombre: str, plugin: Any):
        self.plugins[nombre] = plugin
        self.logger.info(f"Plugin {nombre} registrado")

    async def inicializar(self):
        await self._inicializar_redis()
        await self._inicializar_db()
        self.cargar_modulos()
        self.cargar_plugins()
        for modulo in self.modulos.values():
            await modulo.inicializar(self)
        self.logger.info(f"[CoreCNucleus-{self.instance_id}] Inicializado")

    async def ejecutar(self):
        tasks = [modulo.ejecutar() for modulo in self.modulos.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def detener(self):
        for modulo in self.modulos.values():
            await modulo.detener()
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info(f"[CoreCNucleus-{self.instance_id}] Detenido")

    async def registrar_celu_entidad(self, celu: CeluEntidadCoreC):
        modulo_registro = self.modulos.get("registro")
        if modulo_registro:
            await modulo_registro.registrar_celu_entidad(celu)
        else:
            self.logger.error("Módulo de registro no encontrado")

    async def registrar_micro_celu_entidad(self, micro: MicroCeluEntidadCoreC):
        micro.nucleus = self
        modulo_registro = self.modulos.get("registro")
        if modulo_registro:
            await modulo_registro.registrar_micro_celu_entidad(micro)
        else:
            self.logger.error("Módulo de registro no encontrado")

    async def publicar_alerta(self, alerta: Dict[str, Any]):
        @celery_app.task(autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 5})
        def publicar_alerta_task(alerta_data, db_config, instance_id):
            conn = None
            try:
                conn = psycopg2.connect(**db_config)
                cur = conn.cursor()
                datos_comprimidos = zstd.compress(json.dumps(alerta_data).encode())
                cur.execute(
                    "INSERT INTO bloques (id, canal, num_entidades, fitness, timestamp, instance_id) VALUES (%s, %s, %s, %s, %s, %s)",
                    (f"alerta_{time.time_ns()}", "alertas", 0, 0.0, time.time(), instance_id)
                )
                conn.commit()
                cur.close()
                logging.getLogger("CoreC").info("Alerta publicada")
            except Exception as e:
                logging.getLogger("CoreC").error(f"Error publicando alerta: {e}")
                raise
            finally:
                if conn:
                    conn.close()
        try:
            publicar_alerta_task.delay(alerta, self.db_config, self.instance_id)
        except Exception as e:
            self.logger.error(f"Error encolando alerta: {e}")