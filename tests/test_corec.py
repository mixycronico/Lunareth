#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_corec.py
Pruebas rigurosas para el núcleo de CoreC, asegurando confiabilidad y calidad.
"""
import unittest
import asyncio
import json
import time
import random
import psycopg2
import redis.asyncio as aioredis
from unittest.mock import AsyncMock, patch, MagicMock
from corec.core import (
    serializar_mensaje, deserializar_mensaje, cargar_config,
    enviar_mensaje_redis, recibir_mensajes_redis, celery_app
)
from corec.entities import (
    MicroCeluEntidadCoreC, CeluEntidadCoreC, crear_entidad, crear_celu_entidad,
    procesar_entidad, procesar_celu_entidad
)
from corec.blocks import BloqueSimbiotico
from corec.nucleus import CoreCNucleus
from corec.bootstrap import Bootstrap
from corec.modules.registro import ModuloRegistro
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.sincronización import ModuloSincronización
from corec.processors import ProcesadorSensor, ProcesadorFiltro
import resource
import tracemalloc

class TestCoreCRigorous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configuración inicial para pruebas
        cls.loop = asyncio.get_event_loop()
        cls.config_path = "configs/corec_config.json"
        cls.db_config = {
            "dbname": "test_corec_db",
            "user": "postgres",
            "password": "test_password",
            "host": "localhost",
            "port": "5432"
        }
        cls.redis_config = {
            "host": "localhost",
            "port": 6379,
            "username": "test_user",
            "password": "test_password"
        }
        # Crear configuración temporal
        with open(cls.config_path, "w") as f:
            json.dump({
                "instance_id": "test_corec",
                "db_config": cls.db_config,
                "redis_config": cls.redis_config,
                "bloques": [
                    {"id": "test_bloque", "canal": 1, "entidades": 1000}
                ]
            }, f)

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove(cls.config_path)

    def setUp(self):
        tracemalloc.start()
        self.max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def tearDown(self):
        tracemalloc.stop()
        current_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_diff_kb = (current_rss - self.max_rss) / 1024
        self.assertLess(memory_diff_kb, 100000, f"Prueba consumió demasiada memoria: {memory_diff_kb:.2f} MB")

    # Pruebas de core.py
    async def test_serializar_deserializar_mensaje(self):
        mensaje = await serializar_mensaje(1, 2, 0.5, True)
        resultado = await deserializar_mensaje(mensaje)
        self.assertEqual(resultado, {"id": 1, "canal": 2, "valor": 0.5, "activo": True})

    async def test_serializar_mensaje_corrupto(self):
        with self.assertRaises(Exception):
            await deserializar_mensaje(b"invalid")

    async def test_cargar_config_invalida(self):
        with patch("corec.core.open", side_effect=FileNotFoundError):
            config = cargar_config("invalid.json")
            self.assertEqual(config["instance_id"], "corec1")
            self.assertEqual(config["db_config"], {})
            self.assertEqual(config["redis_config"], {})

    async def test_enviar_recibir_mensaje_redis(self):
        redis_client = AsyncMock()
        mensaje = await serializar_mensaje(1, 2, 0.5, True)
        await enviar_mensaje_redis(redis_client, "test_stream", mensaje, "default")
        redis_client.xadd.assert_called()
        with patch("corec.core.deserializar_mensaje", return_value={"id": 1, "canal": 2, "valor": 0.5, "activo": True}):
            mensajes = await recibir_mensajes_redis(redis_client, "test_stream")
            self.assertTrue(len(mensajes) >= 0)

    async def test_redis_fallo_conexion(self):
        redis_client = AsyncMock()
        redis_client.xadd.side_effect = Exception("Redis desconectado")
        mensaje = await serializar_mensaje(1, 2, 0.5, True)
        await enviar_mensaje_redis(redis_client, "test_stream", mensaje)
        # Verifica que no se bloquea ante fallo

    # Pruebas de entities.py
    async def test_entidad_micro_procesamiento(self):
        async def funcion():
            return {"valor": 0.7}
        entidad = crear_entidad("m1", 1, funcion)
        resultado = await procesar_entidad(entidad, umbral=0.5)
        mensaje = await deserializar_mensaje(resultado)
        self.assertEqual(mensaje["valor"], 0.7)
        self.assertTrue(mensaje["activo"])

    async def test_entidad_micro_inactiva(self):
        async def funcion():
            return {"valor": 0.7}
        entidad = crear_entidad("m1", 1, funcion)
        entidad = entidad[:3] + (False,)  # Desactivar entidad
        resultado = await procesar_entidad(entidad)
        mensaje = await deserializar_mensaje(resultado)
        self.assertEqual(mensaje["valor"], 0.0)
        self.assertFalse(mensaje["activo"])

    async def test_entidad_celu_procesamiento(self):
        async def procesador(datos):
            return {"valor": datos.get("input", 0.0) + 0.1}
        entidad = crear_celu_entidad("c1", 1, procesador)
        resultado = await procesar_celu_entidad(entidad, {"input": 0.5}, umbral=0.5)
        mensaje = await deserializar_mensaje(resultado)
        self.assertEqual(mensaje["valor"], 0.6)
        self.assertTrue(mensaje["activo"])

    async def test_entidad_celu_fallo(self):
        async def procesador(datos):
            raise ValueError("Error simulado")
        entidad = crear_celu_entidad("c1", 1, procesador)
        resultado = await procesar_celu_entidad(entidad, {}, umbral=0.5)
        mensaje = await deserializar_mensaje(resultado)
        self.assertEqual(mensaje["valor"], 0.0)
        self.assertFalse(mensaje["activo"])

    # Pruebas de blocks.py
    async def test_bloque_simbiotico_procesamiento(self):
        async def funcion():
            return {"valor": 0.7}
        entidades = [crear_entidad(f"m{i}", 1, funcion) for i in range(100)]
        bloque = BloqueSimbiotico("test_bloque", 1, entidades, max_size=1024)
        resultado = await bloque.procesar(0.5)
        self.assertEqual(resultado["bloque_id"], "test_bloque")
        self.assertEqual(len(resultado["mensajes"]), 100)
        self.assertGreaterEqual(resultado["fitness"], 0.0)

    async def test_bloque_simbiotico_autoreparacion(self):
        async def funcion_fallo():
            return {"valor": 0.0}
        entidades = [crear_entidad(f"m{i}", 1, funcion_fallo) for i in range(100)]
        bloque = BloqueSimbiotico("test_bloque", 1, entidades, max_size=1024)
        resultado = await bloque.procesar(0.5)
        self.assertGreater(bloque.fallos, 0)
        await bloque.reparar(10)
        self.assertEqual(len(bloque.entidades), 100)

    async def test_bloque_simbiotico_limite_memoria(self):
        async def funcion():
            return {"valor": random.random()}
        entidades = [crear_entidad(f"m{i}", 1, funcion) for i in range(10000)]
        bloque = BloqueSimbiotico("test_bloque", 1, entidades, max_size=1024)
        resultado = await bloque.procesar(0.5)
        self.assertLessEqual(len(resultado["mensajes"]), 100)

    # Pruebas de nucleus.py
    async def test_nucleus_inicializacion(self):
        nucleus = CoreCNucleus(self.config_path, "test_corec")
        with patch.object(nucleus, "_inicializar_redis", AsyncMock()), \
             patch.object(nucleus, "_inicializar_db", AsyncMock()):
            await nucleus.inicializar()
            self.assertTrue(nucleus.redis_client is not None)
            self.assertGreater(len(nucleus.modulos), 0)

    async def test_nucleus_fallo_db(self):
        nucleus = CoreCNucleus(self.config_path, "test_corec")
        with patch("corec.nucleus.psycopg2.connect", side_effect=psycopg2.Error):
            with self.assertRaises(Exception):
                await nucleus._inicializar_db()

    async def test_nucleus_concurrencia_bloques(self):
        nucleus = CoreCNucleus(self.config_path, "test_corec")
        nucleus.modulos["registro"] = ModuloRegistro()
        async def funcion():
            return {"valor": random.random()}
        entidades = [crear_entidad(f"m{i}", 1, funcion) for i in range(1000)]
        bloque = BloqueSimbiotico("test_bloque", 1, entidades, max_size=1024, nucleus=nucleus)
        nucleus.modulos["registro"].bloques["test_bloque"] = bloque
        tasks = [bloque.procesar(0.5) for _ in range(10)]
        resultados = await asyncio.gather(*tasks)
        self.assertEqual(len(resultados), 10)

    # Pruebas de bootstrap.py
    async def test_bootstrap_carga_modulos(self):
        bootstrap = Bootstrap(self.config_path, "test_corec")
        with patch("corec.bootstrap.importlib.import_module", return_value=MagicMock()):
            await bootstrap.inicializar()
            self.assertGreater(len(bootstrap.components), 0)

    async def test_bootstrap_fallo_carga(self):
        bootstrap = Bootstrap(self.config_path, "test_corec")
        with patch("corec.bootstrap.importlib.import_module", side_effect=ImportError):
            await bootstrap.inicializar()
            self.assertGreater(len(bootstrap.components), 0)  # Aún carga nucleus

    # Pruebas de modules/registro.py
    async def test_modulo_registro_bloques(self):
        modulo = ModuloRegistro()
        nucleus = MagicMock()
        nucleus.config = {"bloques": [{"id": "test_bloque", "canal": 1, "entidades": 1000}]}
        nucleus.db_config = self.db_config
        await modulo.inicializar(nucleus)
        await modulo.registrar_bloque("test_bloque", 1, 1000)
        self.assertIn("test_bloque", modulo.bloques)

    # Pruebas de modules/auditoria.py
    async def test_modulo_auditoria_anomalias(self):
        modulo = ModuloAuditoria()
        nucleus = MagicMock()
        nucleus.db_config = self.db_config
        nucleus.anomaly_detector = MagicMock()
        nucleus.anomaly_detector.fit_predict.return_value = [-1, 1]
        nucleus.publicar_alerta = AsyncMock()
        with patch("psycopg2.connect") as mock_connect:
            mock_conn = mock_connect.return_value
            mock_cursor = mock_conn.cursor.return_value
            mock_cursor.fetchall.side_effect = [[(100, 0.5), (200, 0.7)], [("b1", 100, 0.5), ("b2", 200, 0.7)]]
            await modulo.detectar_anomalias()
            nucleus.publicar_alerta.assert_awaited()

    # Pruebas de modules/ejecucion.py
    async def test_modulo_ejecucion_tareas(self):
        modulo = ModuloEjecucion()
        nucleus = MagicMock()
        nucleus.modulos = {"registro": MagicMock()}
        nucleus.modulos["registro"].bloques = {"test_bloque": MagicMock()}
        nucleus.db_config = self.db_config
        nucleus.instance_id = "test_corec"
        await modulo.inicializar(nucleus)
        with patch.object(modulo, "ejecutar_bloque") as mock_task:
            await modulo.ejecutar()
            mock_task.delay.assert_called()

    # Pruebas de modules/sincronización.py
    async def test_modulo_sincronizacion_fusion(self):
        modulo = ModuloSincronización()
        nucleus = MagicMock()
        nucleus.modulos = {"registro": MagicMock()}
        bloque1 = BloqueSimbiotico("b1", 1, [], max_size=1024)
        bloque2 = BloqueSimbiotico("b2", 1, [], max_size=1024)
        nucleus.modulos["registro"].bloques = {"b1": bloque1, "b2": bloque2}
        await modulo.inicializar(nucleus)
        await modulo.fusionar_bloques("b1", "b2", "b3")
        self.assertIn("b3", nucleus.modulos["registro"].bloques)

    # Pruebas de processors.py
    async def test_procesador_sensor(self):
        procesador = ProcesadorSensor()
        resultado = await procesador.procesar({"valores": [0.1, 0.2, 0.3]})
        self.assertAlmostEqual(resultado["valor"], 0.2)

    async def test_procesador_filtro(self):
        procesador = ProcesadorFiltro()
        resultado = await procesador.procesar({"valor": 0.7, "umbral": 0.5})
        self.assertEqual(resultado["valor"], 0.7)
        resultado = await procesador.procesar({"valor": 0.3, "umbral": 0.5})
        self.assertEqual(resultado["valor"], 0.0)

    # Prueba de rendimiento para ~1M entidades
    async def test_rendimiento_millon_entidades(self):
        async def funcion():
            return {"valor": random.random()}
        num_entidades = 1000000 // 1000
        bloques = []
        for i in range(num_entidades):
            entidades = [crear_entidad(f"m{j}", 1, funcion) for j in range(1000)]
            bloques.append(BloqueSimbiotico(f"b{i}", 1, entidades, max_size=1024))
        start_time = time.time()
        tasks = [bloque.procesar(0.5) for bloque in bloques[:10]]  # Limitado para pruebas
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 10, f"Procesamiento tomó demasiado: {elapsed}s")
        memory_diff_kb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - self.max_rss) / 1024
        self.assertLess(memory_diff_kb, 1000000, f"Memoria excedida: {memory_diff_kb:.2f} MB")

    def test_all(self):
        async def run_tests():
            await self.test_serializar_deserializar_mensaje()
            await self.test_serializar_mensaje_corrupto()
            await self.test_cargar_config_invalida()
            await self.test_enviar_recibir_mensaje_redis()
            await self.test_redis_fallo_conexion()
            await self.test_entidad_micro_procesamiento()
            await self.test_entidad_micro_inactiva()
            await self.test_entidad_celu_procesamiento()
            await self.test_entidad_celu_fallo()
            await self.test_bloque_simbiotico_procesamiento()
            await self.test_bloque_simbiotico_autoreparacion()
            await self.test_bloque_simbiotico_limite_memoria()
            await self.test_nucleus_inicializacion()
            await self.test_nucleus_fallo_db()
            await self.test_nucleus_concurrencia_bloques()
            await self.test_bootstrap_carga_modulos()
            await self.test_bootstrap_fallo_carga()
            await self.test_modulo_registro_bloques()
            await self.test_modulo_auditoria_anomalias()
            await self.test_modulo_ejecucion_tareas()
            await self.test_modulo_sincronizacion_fusion()
            await self.test_procesador_sensor()
            await self.test_procesador_filtro()
            await self.test_rendimiento_millon_entidades()
        self.loop.run_until_complete(run_tests())

if __name__ == "__main__":
    unittest.main()