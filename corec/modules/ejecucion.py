#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corec/modules/ejecucion.py
Módulo de ejecución de tareas Celery en CoreC.
"""
from corec.core import ModuloBase, celery_app, asyncio, logging, psycopg2
from typing import Dict, Any

class ModuloEjecucion(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloEjecucion")
        self.nucleus = None

    async def inicializar(self, nucleus):
        self.nucleus = nucleus
        self.logger.info("[ModuloEjecucion] Inicializado")

    @celery_app.task(autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 5})
    def ejecutar_bloque(bloque_id: str, db_config: Dict[str, Any], instance_id: str):
        from corec.blocks import BloqueSimbiotico
        try:
            conn = psycopg2.connect(**db_config)
            try:
                cur = conn.cursor()
                cur.execute("SELECT canal, num_entidades FROM bloques WHERE id = %s", (bloque_id,))
                result = cur.fetchone()
                if result:
                    canal, num_entidades = result
                    bloque = BloqueSimbiotico(bloque_id, canal, [], max_size=1024)
                    asyncio.run(bloque.escribir_postgresql(db_config))
                    logging.getLogger("ModuloEjecucion").info(f"Bloque {bloque_id} ejecutado")
                else:
                    logging.getLogger("ModuloEjecucion").error(f"Bloque {bloque_id} no encontrado")
                cur.close()
            finally:
                conn.close()
        except Exception as e:
            logging.getLogger("ModuloEjecucion").error(f"Error ejecutando bloque {bloque_id}: {e}")
            raise

    async def ejecutar(self):
        while True:
            try:
                modulo_registro = self.nucleus.modulos.get("registro")
                if modulo_registro:
                    for bloque_id in modulo_registro.bloques.keys():
                        self.ejecutar_bloque.delay(bloque_id, self.nucleus.db_config, self.nucleus.instance_id)
                    self.logger.info("Tareas de bloques encoladas")
            except Exception as e:
                self.logger.error(f"Error encolando tareas: {e}")
            await asyncio.sleep(60)

    async def detener(self):
        self.logger.info("[ModuloEjecucion] Detenido")