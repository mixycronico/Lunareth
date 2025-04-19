from corec.core import IsolationForest, psycopg2, zstd, json, statistics, logging, asyncio, time, random, serializar_mensaje, deserializar_mensaje
from corec.entities import MicroCeluEntidadCoreC, crear_entidad, procesar_entidad
from typing import Dict, Any, List

class BloqueSimbiotico:
    def __init__(self, id: str, canal: int, entidades: list[MicroCeluEntidadCoreC], max_size: int = 1024, nucleus=None):
        self.id = id
        self.canal = canal
        self.entidades = entidades[:max_size // 1]
        self.fitness = 0.0
        self.nucleus = nucleus
        self.logger = logging.getLogger(f"BloqueSimbiotico-{id}")
        self.anomaly_detector = IsolationForest(contamination=0.05)
        self.mensajes = []
        self.umbral = 0.5
        self.fallos = 0

    async def ajustar_umbral(self, carga: float, valores: list[float], errores: int):
        try:
            desviacion = statistics.stdev(valores) if len(valores) > 1 else 0.1
            self.umbral = min(max(0.5 * carga + 0.3 * desviacion + 0.2 * (errores / len(self.entidades)), 0.1), 0.9)
            self.logger.info(f"Bloque {self.id} ajustó umbral a {self.umbral:.2f}")
        except Exception as e:
            self.logger.error(f"[BloqueSimbiotico {self.id}] Error ajustando umbral: {e}")

    async def procesar(self, carga: float) -> Dict[str, Any]:
        num_entidades = int(len(self.entidades) * min(carga, 1.0)) or 1
        entidades_activas = self.entidades[:num_entidades]
        resultados = await asyncio.gather(*(procesar_entidad(e, self.umbral) for e in entidades_activas), return_exceptions=True)
        mensajes = []
        errores = 0
        valores = []
        for r in resultados:
            mensaje = await deserializar_mensaje(r)
            mensajes.append(mensaje)
            if not mensaje["activo"]:
                errores += 1
            if mensaje["valor"] > 0:
                valores.append(mensaje["valor"])
        self.fitness = max(0.0, self.fitness - errores / num_entidades)
        self.mensajes.extend(mensajes)
        await self.ajustar_umbral(carga, valores, errores)
        if errores > num_entidades * 0.05:
            await self.reparar(errores)
        return {"bloque_id": self.id, "mensajes": mensajes[:100], "fitness": self.fitness}

    async def reparar(self, errores: int):
        for i, mensaje in enumerate(self.mensajes):
            if not mensaje["activo"]:
                self.fallos += 1
                if self.fallos >= 2:
                    self.entidades[i] = crear_entidad(f"m{time.time_ns()}", self.canal, self.entidades[i][2])
                    self.fallos = 0
                else:
                    entidad_activa = next((e for e in self.entidades if e[3]), self.entidades[i])
                    self.entidades[i] = crear_entidad(f"m{time.time_ns()}", self.canal, entidad_activa[2])
        self.logger.info(f"Bloque {self.id} reparado: {errores} entidades reemplazadas")
        self.mensajes.clear()

    async def escribir_postgresql(self, db_config: Dict[str, Any]):
        resultados = await self.procesar(0.5)
        try:
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            datos_comprimidos = zstd.compress(json.dumps(resultados["mensajes"]).encode(), level=3)
            cur.execute(
                "INSERT INTO bloques (id, canal, num_entidades, fitness, timestamp, instance_id) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET num_entidades = %s, fitness = %s, timestamp = %s",
                (self.id, self.canal, len(self.entidades), self.fitness, time.time(), self.nucleus.instance_id,
                 len(self.entidades), self.fitness, time.time())
            )
            conn.commit()
            cur.close()
            conn.close()
            self.mensajes.clear()
        except Exception as e:
            self.logger.error(f"[BloqueSimbiotico {self.id}] Error escribiendo en PostgreSQL: {e}")