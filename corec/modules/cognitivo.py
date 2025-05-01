import time
import json
from typing import Dict, Any, List
from corec.core import ComponenteBase


class ModuloCognitivo(ComponenteBase):
    """Módulo cognitivo con autoconciencia avanzada, atención y resolución de conflictos."""
    def __init__(self):
        self.nucleus = None
        self.logger = None
        self.memoria: Dict[str, List[Dict[str, Any]]] = {}
        self.intuiciones: Dict[str, float] = {}
        self.percepciones: List[Dict[str, Any]] = []
        self.decisiones: List[Dict[str, Any]] = []
        self.decisiones_fallidas: List[Dict[str, Any]] = []
        self.contexto: Dict[str, Any] = {}
        self.memoria_semantica: Dict[str, Dict[str, float]] = {}
        self.intenciones: List[Dict[str, Any]] = []
        self.atencion: Dict[str, Any] = {
            "focos": [],
            "nivel": 0.5
        }
        self.yo: Dict[str, Any] = {
            "estado": {
                "confianza": 1.0,
                "estabilidad": 1.0,
                "actividad": 0.0
            },
            "memoria": {
                "conceptos": {},
                "tamaño": 0
            }
        }
        self.ultima_evaluacion = time.time()
        self.memoria_semantica_old = {}
        self.conflictos_intenciones = {
            ("aumentar_percepciones", "reducir_umbral_confianza"): "aumentar actividad vs. reducir errores"
        }
        self.config = None

    async def inicializar(self, nucleus, config: Dict[str, Any] = None):
        """Inicializa el módulo cognitivo."""
        try:
            self.nucleus = nucleus
            self.logger = nucleus.logger
            self.config = config or {
                "max_memoria": 1000,
                "umbral_confianza": 0.5,
                "penalizacion_intuicion": 0.9,
                "max_percepciones": 5000,
                "impacto_adaptacion": 0.1,
                "confiabilidad_minima": 0.4,
                "umbral_afectivo_positivo": 0.8,
                "umbral_afectivo_negativo": -0.8,
                "peso_afectivo": 0.2,
                "umbral_fallo": 0.3,
                "peso_semantico": 0.1,
                "umbral_cambio_significativo": 0.05,
                "tasa_aprendizaje_minima": 0.1,
                "umbral_relevancia": 0.3,
                "peso_novedad": 0.3
            }
            for key, value in self.config.items():
                if key in ["max_memoria", "max_percepciones"] and value <= 0:
                    raise ValueError(f"{key} debe ser mayor que 0")
                if key in ["umbral_confianza", "penalizacion_intuicion", "confiabilidad_minima",
                           "umbral_fallo", "tasa_aprendizaje_minima", "umbral_relevancia"] and not 0 < value <= 1:
                    raise ValueError(f"{key} debe estar entre 0 y 1")
                if key in ["impacto_adaptacion", "peso_afectivo", "peso_semantico",
                           "umbral_cambio_significativo", "peso_novedad"] and not 0 <= value <= 1:
                    raise ValueError(f"{key} debe estar entre 0 y 1")

            await self.cargar_estado()
            await self.actualizar_yo()
            await self.actualizar_atencion()
            self.logger.info("Módulo Cognitivo inicializado")
            await self.nucleus.publicar_alerta({
                "tipo": "cognitivo_inicializado",
                "mensaje": "Módulo cognitivo inicializado",
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"Error inicializando Módulo Cognitivo: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_cognitivo",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def cargar_estado(self):
        """Carga el estado desde PostgreSQL."""
        try:
            if not self.nucleus.db_pool:
                self.logger.warning("No hay conexión a PostgreSQL, iniciando con estado vacío")
                return

            async with self.nucleus.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT memoria, intuiciones, percepciones, decisiones, decisiones_fallidas, contexto, memoria_semantica, yo, intenciones, atencion
                    FROM cognitivo_memoria
                    WHERE instancia_id = $1
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    self.nucleus.config.instance_id
                )
                if row:
                    self.memoria = json.loads(row["memoria"]) if row["memoria"] else {}
                    self.intuiciones = json.loads(row["intuiciones"]) if row["intuiciones"] else {}
                    self.percepciones = json.loads(row["percepciones"]) if row["percepciones"] else []
                    self.decisiones = json.loads(row["decisiones"]) if row["decisiones"] else []
                    self.decisiones_fallidas = json.loads(row["decisiones_fallidas"]) if row["decisiones_fallidas"] else []
                    self.contexto = json.loads(row["contexto"]) if row["contexto"] else {}
                    self.memoria_semantica = json.loads(row["memoria_semantica"]) if row["memoria_semantica"] else {}
                    self.yo = json.loads(row["yo"]) if row["yo"] else self.yo
                    self.intenciones = json.loads(row["intenciones"]) if row["intenciones"] else []
                    self.atencion = json.loads(row["atencion"]) if row["atencion"] else self.atencion
                    self.logger.info("Estado cognitivo cargado desde PostgreSQL")
                else:
                    self.logger.info("No se encontró estado previo, iniciando con estado vacío")
        except Exception as e:
            self.logger.error(f"Error cargando estado cognitivo: {e}")

    async def guardar_estado(self):
        """Guarda el estado en PostgreSQL, limitando datos persistidos."""
        try:
            if not self.nucleus.db_pool:
                self.logger.warning("No hay conexión a PostgreSQL, no se puede guardar el estado")
                return

            max_percepciones = 100  # Limitar percepciones guardadas
            max_decisiones = 100  # Limitar decisiones guardadas
            percepciones = self.percepciones[-max_percepciones:]
            decisiones = self.decisiones[-max_decisiones:]

            async with self.nucleus.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO cognitivo_memoria (
                        instancia_id, tipo, memoria, intuiciones, percepciones, decisiones, decisiones_fallidas, contexto, memoria_semantica, yo, intenciones, atencion, timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    self.nucleus.config.instance_id,
                    "estado_completo",
                    json.dumps(self.memoria),
                    json.dumps(self.intuiciones),
                    json.dumps(percepciones),
                    json.dumps(decisiones),
                    json.dumps(self.decisiones_fallidas[-100:]),
                    json.dumps(self.contexto),
                    json.dumps(self.memoria_semantica),
                    json.dumps(self.yo),
                    json.dumps(self.intenciones),
                    json.dumps(self.atencion),
                    time.time()
                )
            self.logger.info("Estado cognitivo guardado en PostgreSQL")
        except Exception as e:
            self.logger.error(f"Error guardando estado cognitivo: {e}")

    async def actualizar_atencion(self):
        """Actualiza el sistema de atención cognitiva."""
        try:
            self.atencion["focos"] = [i["meta"] for i in self.intenciones if i["estado"] == "activa"]
            for concepto in self.memoria_semantica.get("yo", {}):
                if self.memoria_semantica["yo"].get(concepto, 0.0) > 0.5:
                    self.atencion["focos"].append(concepto)
            self.atencion["nivel"] = 0.5 + (1.0 - self.yo["estado"]["actividad"] / 20) if self.atencion["focos"] else 0.3
            self.atencion["nivel"] = min(self.atencion["nivel"], 1.0)

            if self.nucleus.db_pool:
                async with self.nucleus.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO cognitivo_atencion (instancia_id, atencion, timestamp)
                        VALUES ($1, $2, $3)
                        """,
                        self.nucleus.config.instance_id,
                        json.dumps(self.atencion),
                        time.time()
                    )
            await self.nucleus.publicar_alerta({
                "tipo": "cognitivo_atencion",
                "atencion": self.atencion,
                "timestamp": time.time()
            })
            self.logger.debug(f"Atención actualizada: {self.atencion}")
        except Exception as e:
            self.logger.error(f"Error actualizando atención: {e}")

    async def evaluar_relevancia(self, datos: Dict[str, Any]) -> float:
        """Evalúa la relevancia de una percepción."""
        try:
            clave = datos.get("tipo", "desconocido")
            relevancia = 0.0
            if clave in self.atencion["focos"]:
                relevancia += 0.5
            for foco in self.atencion["focos"]:
                if foco in self.memoria_semantica and clave in self.memoria_semantica[foco]:
                    relevancia += self.memoria_semantica[foco][clave] * 0.3
            impacto = abs(datos.get("impacto_afectivo", 0.0))
            relevancia += impacto * 0.2
            novedad = 1.0 / (1 + (time.time() - datos["timestamp"]) / 3600)
            relevancia += novedad * self.config.get("peso_novedad", 0.3)
            return min(relevancia, 1.0)
        except Exception as e:
            self.logger.error(f"Error evaluando relevancia: {e}")
            return 0.0

    async def actualizar_yo(self):
        """Actualiza el modelo interno del 'yo'."""
        try:
            confiabilidad = await self.evaluar_confiabilidad()
            fallos_recientes = len([d for d in self.decisiones_fallidas[-10:]])
            decisiones_recientes = len([d for d in self.decisiones[-10:]])
            estabilidad = 1.0 - (fallos_recientes / decisiones_recientes) if decisiones_recientes else 1.0
            actividad = (
                len(self.percepciones[-100:]) /
                (time.time() - self.percepciones[-100]["timestamp"] + 1e-6)
                if self.percepciones and len(self.percepciones) >= 100 else 0.0
            )

            umbral_cambio = self.config.get("umbral_cambio_significativo", 0.05)
            if abs(self.yo["estado"]["confianza"] - confiabilidad) > umbral_cambio:
                await self.registrar_cambio_yo(
                    "estado.confianza",
                    self.yo["estado"]["confianza"],
                    confiabilidad,
                    "actualización de confiabilidad"
                )
            if abs(self.yo["estado"]["estabilidad"] - estabilidad) > umbral_cambio:
                await self.registrar_cambio_yo(
                    "estado.estabilidad",
                    self.yo["estado"]["estabilidad"],
                    estabilidad,
                    "actualización de estabilidad"
                )

            self.yo["estado"] = {
                "confianza": confiabilidad,
                "estabilidad": estabilidad,
                "actividad": actividad
            }
            self.yo["memoria"]["tamaño"] = len(self.percepciones)
            self.yo["memoria"]["conceptos"] = self.memoria_semantica.get("yo", {})
            self.logger.debug(f"Modelo interno actualizado: {self.yo}")
            await self.generar_intenciones()
            await self.actualizar_atencion()
        except Exception as e:
            self.logger.error(f"Error actualizando modelo interno: {e}")

    async def registrar_cambio_yo(self, atributo: str, valor_anterior: Any, valor_nuevo: Any, motivo: str):
        """Registra un cambio en el modelo interno."""
        try:
            cambio = {
                "atributo": atributo,
                "valor_anterior": valor_anterior,
                "valor_nuevo": valor_nuevo,
                "motivo": motivo,
                "timestamp": time.time()
            }
            if self.nucleus.redis_client:
                await self.nucleus.redis_client.xadd("corec:cognitivo:cambios_yo", cambio, maxlen=1000)
            if self.nucleus.db_pool:
                async with self.nucleus.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO cognitivo_cambios_yo (instancia_id, cambio, timestamp)
                        VALUES ($1, $2, $3)
                        """,
                        self.nucleus.config.instance_id,
                        json.dumps(cambio),
                        cambio["timestamp"]
                    )
            self.logger.debug(f"Cambio en yo registrado: {cambio}")
        except Exception as e:
            self.logger.error(f"Error registrando cambio en yo: {e}")

    async def generar_intenciones(self):
        """Genera intenciones explícitas."""
        try:
            intenciones = []
            if self.yo["estado"]["estabilidad"] < 0.7:
                intenciones.append({
                    "meta": "mejorar_estabilidad",
                    "prioridad": 0.8,
                    "condicion": {"estabilidad": {"max": 0.7}},
                    "acciones": ["reducir_umbral_confianza", "analizar_fallos"],
                    "estado": "activa",
                    "tipo": "intencion",
                    "timestamp": time.time()
                })
            tasa_aprendizaje = await self.evaluar_aprendizaje()
            if tasa_aprendizaje < self.config.get("tasa_aprendizaje_minima", 0.1):
                intenciones.append({
                    "meta": "aprender_mejor",
                    "prioridad": 0.6,
                    "condicion": {"tasa_aprendizaje": {"max": 0.1}},
                    "acciones": ["aumentar_percepciones", "reforzar_memoria_semantica"],
                    "estado": "activa",
                    "tipo": "objetivo",
                    "timestamp": time.time()
                })

            for intencion in intenciones:
                if intencion not in self.intenciones:
                    self.intenciones.append(intencion)
                    if self.nucleus.db_pool:
                        async with self.nucleus.db_pool.acquire() as conn:
                            await conn.execute(
                                """
                                INSERT INTO cognitivo_intenciones (instancia_id, intencion, timestamp)
                                VALUES ($1, $2, $3)
                                """,
                                self.nucleus.config.instance_id,
                                json.dumps(intencion),
                                intencion["timestamp"]
                            )
                    await self.nucleus.publicar_alerta({
                        "tipo": "cognitivo_intencion",
                        "intencion": intencion,
                        "timestamp": intencion["timestamp"]
                    })
            self.logger.info(f"Intenciones generadas: {[i['meta'] for i in intenciones]}")
            await self.resolver_conflictos()
            await self.ejecutar_intenciones()
        except Exception as e:
            self.logger.error(f"Error generando intenciones: {e}")

    async def resolver_conflictos(self):
        """Resuelve conflictos entre intenciones."""
        try:
            conflictos = []
            for i1 in self.intenciones:
                for i2 in self.intenciones:
                    if i1 != i2 and i1["estado"] == "activa" and i2["estado"] == "activa":
                        for a1 in i1["acciones"]:
                            for a2 in i2["acciones"]:
                                if (a1, a2) in self.conflictos_intenciones:
                                    conflictos.append((i1, i2, self.conflictos_intenciones[(a1, a2)]))
            for i1, i2, motivo in conflictos:
                resolucion = {
                    "intencion1": i1["meta"],
                    "intencion2": i2["meta"],
                    "motivo": motivo
                }
                if self.yo["estado"]["estabilidad"] < 0.7:
                    resolucion["decision"] = f"Priorizar {i1['meta']} debido a estabilidad baja"
                    i2["estado"] = "desactivada"
                else:
                    resolucion["decision"] = "Compromiso: reducir umbral parcialmente"
                    self.config["umbral_confianza"] *= 0.95
                if self.nucleus.db_pool:
                    async with self.nucleus.db_pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO cognitivo_conflictos (instancia_id, conflicto, timestamp)
                            VALUES ($1, $2, $3)
                            """,
                            self.nucleus.config.instance_id,
                            json.dumps(resolucion),
                            time.time()
                        )
                await self.nucleus.publicar_alerta({
                    "tipo": "cognitivo_conflicto",
                    "resolucion": resolucion,
                    "timestamp": time.time()
                })
                self.logger.info(f"Conflicto resuelto: {resolucion['decision']}")
            self.intenciones = [i for i in self.intenciones if i["estado"] == "activa"]
        except Exception as e:
            self.logger.error(f"Error resolviendo conflictos: {e}")

    async def ejecutar_intenciones(self):
        """Ejecuta acciones de intenciones activas."""
        try:
            for intencion in self.intenciones:
                if intencion["estado"] != "activa":
                    continue
                for accion in intencion["acciones"]:
                    if accion == "reducir_umbral_confianza":
                        self.config["umbral_confianza"] = max(self.config["umbral_confianza"] * 0.9, 0.3)
                        self.logger.info(
                            f"Umbral de confianza reducido a {self.config['umbral_confianza']:.2f}"
                        )
                    elif accion == "analizar_fallos":
                        await self.analizar_decisiones_fallidas()
                    elif accion == "aumentar_percepciones":
                        self.logger.info("Solicitando más percepciones...")
                    elif accion == "reforzar_memoria_semantica":
                        for concepto in self.memoria_semantica:
                            for otro_concepto in self.memoria_semantica[concepto]:
                                self.memoria_semantica[concepto][otro_concepto] *= 1.05
                                self.memoria_semantica[concepto][otro_concepto] = min(
                                    self.memoria_semantica[concepto][otro_concepto], 1.0
                                )
                        self.logger.info("Memoria semántica reforzada")
                if intencion["meta"] == "mejorar_estabilidad" and self.yo["estado"]["estabilidad"] >= 0.7:
                    intencion["estado"] = "completada"
                elif intencion["meta"] == "aprender_mejor" and (await self.evaluar_aprendizaje()) >= 0.1:
                    intencion["estado"] = "completada"
            self.intenciones = [i for i in self.intenciones if i["estado"] == "activa"]
        except Exception as e:
            self.logger.error(f"Error ejecutando intenciones: {e}")

    async def evaluar_aprendizaje(self):
        """Evalúa la tasa de aprendizaje."""
        try:
            conceptos_nuevos = len([c for c in self.memoria_semantica if c not in self.memoria_semantica_old])
            tasa_aprendizaje = conceptos_nuevos / (time.time() - self.ultima_evaluacion + 1e-6)
            self.memoria_semantica_old = self.memoria_semantica.copy()
            self.ultima_evaluacion = time.time()
            self.logger.debug(f"Tasa de aprendizaje: {tasa_aprendizaje:.4f}")
            return tasa_aprendizaje
        except Exception as e:
            self.logger.error(f"Error evaluando aprendizaje: {e}")
            return 0.0

    async def consultar_metadialogos_previos(self, tema: str, max_edad: float = 3600) -> List[Dict[str, Any]]:
        """Consulta metadialogos previos."""
        try:
            if not self.nucleus.db_pool:
                return []
            async with self.nucleus.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT afirmacion, contexto, timestamp
                    FROM cognitivo_metadialogo
                    WHERE instancia_id = $1 AND afirmacion ILIKE $2 AND timestamp > $3
                    ORDER BY timestamp DESC LIMIT 5
                    """,
                    self.nucleus.config.instance_id,
                    f"%{tema}%",
                    time.time() - max_edad
                )
            return [
                {
                    "afirmacion": row["afirmacion"],
                    "contexto": json.loads(row["contexto"]),
                    "timestamp": row["timestamp"]
                } for row in rows
            ]
        except Exception as e:
            self.logger.error(f"Error consultando metadialogos previos: {e}")
            return []

    async def generar_metadialogo(self):
        """Genera afirmaciones sobre el estado interno."""
        try:
            afirmaciones = []
            if self.yo["estado"]["confianza"] < 0.5:
                afirmacion = "Creo que mi razonamiento es poco confiable porque mi confianza es baja."
                previos = await self.consultar_metadialogos_previos("confiable")
                if previos:
                    afirmacion += (
                        f" Esto es consistente con mi reflexión anterior: '{previos[0]['afirmacion']}'."
                    )
                afirmaciones.append({"texto": afirmacion, "referencias": [p["timestamp"] for p in previos]})
            fallos_recientes = len([d for d in self.decisiones_fallidas[-10:]])
            if self.yo["estado"]["estabilidad"] < 0.7 and fallos_recientes > 2:
                afirmacion = f"Me siento inestable porque fallé {fallos_recientes} veces seguidas."
                previos = await self.consultar_metadialogos_previos("inestable")
                if previos:
                    afirmacion += f" Esto sigue un patrón desde: '{previos[0]['afirmacion']}'."
                afirmaciones.append({"texto": afirmacion, "referencias": [p["timestamp"] for p in previos]})
            if self.yo["estado"]["actividad"] > 10:
                afirmaciones.append({
                    "texto": "Estoy muy activo, procesando muchas percepciones.",
                    "referencias": []
                })
            contradicciones = await self.detectar_contradicciones()
            if contradicciones:
                afirmacion = "Detecté contradicciones en mis intuiciones, debo revisar mis conceptos."
                afirmaciones.append({"texto": afirmacion, "referencias": []})

            for intencion in self.intenciones:
                afirmacion = f"Quiero {intencion['meta']} porque {intencion['condicion']}"
                afirmaciones.append({"texto": afirmacion, "referencias": []})

            for a in afirmaciones:
                if self.nucleus.db_pool:
                    async with self.nucleus.db_pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO cognitivo_metadialogo (instancia_id, afirmacion, contexto, referencias, timestamp)
                            VALUES ($1, $2, $3, $4, $5)
                            """,
                            self.nucleus.config.instance_id,
                            a["texto"],
                            json.dumps(self.yo),
                            json.dumps(a["referencias"]),
                            time.time()
                        )
                await self.nucleus.publicar_alerta({
                    "tipo": "cognitivo_metadialogo",
                    "afirmacion": a["texto"],
                    "referencias": a["referencias"],
                    "contexto": self.yo,
                    "timestamp": time.time()
                })
            self.logger.info(f"Metadialogo generado: {[a['texto'] for a in afirmaciones]}")
            return [a["texto"] for a in afirmaciones]
        except Exception as e:
            self.logger.error(f"Error generando metadialogo: {e}")
            return []

    async def percibir(self, datos: Dict[str, Any]):
        """Recibe datos del entorno con atención selectiva."""
        try:
            if not isinstance(datos, dict) or "tipo" not in datos:
                raise ValueError("Datos inválidos: deben ser un diccionario con clave 'tipo'")

            datos = datos.copy()
            datos["timestamp"] = datos.get("timestamp", time.time())
            valor = datos.get("valor", 0.0)
            if isinstance(valor, (int, float)):
                umbral_positivo = self.config.get("umbral_afectivo_positivo", 0.8)
                umbral_negativo = self.config.get("umbral_afectivo_negativo", -0.8)
                if valor >= umbral_positivo:
                    datos["impacto_afectivo"] = 0.8
                elif valor <= umbral_negativo:
                    datos["impacto_afectivo"] = -0.8
                else:
                    datos["impacto_afectivo"] = 0.0
            else:
                datos["impacto_afectivo"] = 0.0

            relevancia = await self.evaluar_relevancia(datos)
            if relevancia < self.config.get("umbral_relevancia", 0.3) and self.atencion["nivel"] > 0.7:
                self.logger.debug(f"Percepción descartada por baja relevancia: {datos['tipo']}")
                return

            self.percepciones.append(datos)
            clave = datos["tipo"]

            if clave not in self.memoria:
                self.memoria[clave] = []
            self.memoria[clave].append(datos)

            max_memoria = self.config.get("max_memoria", 1000)
            if len(self.memoria[clave]) > max_memoria:
                self.memoria[clave] = self.memoria[clave][-max_memoria:]

            max_percepciones = self.config.get("max_percepciones", 5000)
            if len(self.percepciones) > max_percepciones:
                self.percepciones = self.percepciones[-max_percepciones:]

            await self.aprender_concepto(clave, datos)
            await self.actualizar_yo()
            self.logger.debug(
                f"Percepción registrada: {clave}, impacto afectivo: {datos['impacto_afectivo']:.2f}"
            )
            await self.nucleus.publicar_alerta({
                "tipo": "cognitivo_percepcion",
                "clave": clave,
                "impacto_afectivo": datos["impacto_afectivo"],
                "relevancia": relevancia,
                "timestamp": datos["timestamp"]
            })
        except Exception as e:
            self.logger.error(f"Error procesando percepción: {e}")

    async def aprender_concepto(self, concepto: str, datos: Dict[str, Any]):
        """Aprende relaciones semánticas."""
        try:
            if concepto not in self.memoria_semantica:
                self.memoria_semantica[concepto] = {}
            if "yo" not in self.memoria_semantica:
                self.memoria_semantica["yo"] = {}

            recientes = self.percepciones[-5:]
            for p in recientes:
                otro_concepto = p.get("tipo")
                if otro_concepto != concepto and otro_concepto:
                    peso = self.memoria_semantica[concepto].get(otro_concepto, 0.0) + 0.1
                    self.memoria_semantica[concepto][otro_concepto] = min(peso, 1.0)
                    peso_yo = self.memoria_semantica["yo"].get(concepto, 0.0) + 0.05
                    self.memoria_semantica["yo"][concepto] = min(peso_yo, 1.0)
                    self.logger.debug(
                        f"Relación semántica aprendida: {concepto} -> {otro_concepto} ({peso:.2f})"
                    )
        except Exception as e:
            self.logger.error(f"Error aprendiendo concepto {concepto}: {e}")

    async def consultar_memoria_semantica(self, concepto: str) -> Dict[str, float]:
        """Consulta relaciones semánticas."""
        return self.memoria_semantica.get(concepto, {})

    async def intuir(self, tipo: str) -> float:
        """Genera una intuición."""
        try:
            if not isinstance(tipo, str) or not tipo:
                raise ValueError("El tipo debe ser una cadena no vacía")

            historial = self.memoria.get(tipo, [])
            if not historial:
                return 0.0

            valores = [float(d.get("valor", 0.0)) for d in historial if isinstance(d.get("valor"), (int, float))]
            impactos = [
                float(d.get("impacto_afectivo", 0.0))
                for d in historial if isinstance(d.get("valor"), (int, float))
            ]
            if not valores:
                return 0.0

            pesos = [
                1.0 / (1 + (time.time() - d["timestamp"]) / 3600)
                for d in historial if isinstance(d.get("valor"), (int, float))
            ]
            peso_afectivo = self.config.get("peso_afectivo", 0.2)
            valores_ajustados = [v + i * peso_afectivo for v, i in zip(valores, impactos)]
            intuicion_base = sum(v * w for v, w in zip(valores_ajustados, pesos)) / sum(pesos) if sum(pesos) > 0 else 0.0

            conceptos_relacionados = await self.consultar_memoria_semantica(tipo)
            ajuste_semantico = sum(
                self.intuiciones.get(c, 0.0) * peso for c, peso in conceptos_relacionados.items()
            )
            intuicion = intuicion_base + ajuste_semantico * self.config.get("peso_semantico", 0.1)
            self.intuiciones[tipo] = intuicion
            self.logger.debug(f"Intuición generada para {tipo}: {intuicion:.4f}")
            return intuicion
        except Exception as e:
            self.logger.error(f"Error generando intuición para {tipo}: {e}")
            return 0.0

    async def decidir(self, opciones: List[str], umbral: float = None) -> str:
        """Toma una decisión."""
        try:
            if not isinstance(opciones, list) or not opciones:
                raise ValueError("Las opciones deben ser una lista no vacía")

            umbral = umbral if umbral is not None else self.config.get("umbral_confianza", 0.5)
            mejor_opcion = "ninguna"
            max_confianza = 0.0

            for opcion in opciones:
                confianza = await self.intuir(opcion)
                if opcion in self.contexto:
                    confianza += self.contexto[opcion].get("impacto_afectivo", 0.0) * self.config.get("peso_afectivo", 0.2)
                if confianza > max_confianza and confianza >= umbral:
                    mejor_opcion = opcion
                    max_confianza = confianza

            decision = {
                "opcion": mejor_opcion,
                "confianza": max_confianza,
                "timestamp": time.time()
            }
            self.decisiones.append(decision)
            if max_confianza < self.config.get("umbral_fallo", 0.3):
                await self.registrar_decision_fallida(decision, motivo="baja_confianza")

            await self.actualizar_yo()
            self.logger.info(f"Decisión tomada: {mejor_opcion} con confianza {max_confianza:.2f}")
            return mejor_opcion
        except Exception as e:
            self.logger.error(f"Error tomando decisión: {e}")
            return "ninguna"

    async def registrar_decision_fallida(self, decision: Dict[str, Any], motivo: str):
        """Registra una decisión fallida."""
        try:
            decision_fallida = {
                "opcion": decision["opcion"],
                "confianza": decision["confianza"],
                "motivo": motivo,
                "contexto": self.contexto,
                "timestamp": decision["timestamp"]
            }
            self.decisiones_fallidas.append(decision_fallida)
            if self.nucleus.db_pool:
                async with self.nucleus.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO cognitivo_decisiones_fallidas (
                            instancia_id, opcion, confianza, motivo_fallo, contexto, timestamp
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        self.nucleus.config.instance_id,
                        decision_fallida["opcion"],
                        decision_fallida["confianza"],
                        motivo,
                        json.dumps(self.contexto),
                        decision_fallida["timestamp"]
                    )
            self.logger.warning(f"Decisión fallida registrada: {decision_fallida['opcion']} ({motivo})")
        except Exception as e:
            self.logger.error(f"Error registrando decisión fallida: {e}")

    async def analizar_decisiones_fallidas(self):
        """Analiza decisiones fallidas para ajustar intuiciones."""
        try:
            if not self.decisiones_fallidas:
                self.logger.debug("No hay decisiones fallidas para analizar")
                return

            fallos_por_opcion = {}
            for decision in self.decisiones_fallidas[-50:]:
                opcion = decision["opcion"]
                fallos_por_opcion[opcion] = fallos_por_opcion.get(opcion, 0) + 1

            for opcion, count in fallos_por_opcion.items():
                if count >= 3:
                    if opcion in self.intuiciones:
                        self.intuiciones[opcion] *= 0.8
                        self.logger.warning(
                            f"Intuición penalizada para {opcion} por {count} fallos: {self.intuiciones[opcion]:.4f}"
                        )
        except Exception as e:
            self.logger.error(f"Error analizando decisiones fallidas: {e}")

    async def evaluar_confiabilidad(self):
        """Evalúa la confiabilidad histórica."""
        try:
            if not self.decisiones:
                return 1.0

            confiabilidades = [d["confianza"] for d in self.decisiones[-10:]]
            confiabilidad = sum(confiabilidades) / len(confiabilidades) if confiabilidades else 1.0
            confiabilidad_minima = self.config.get("confiabilidad_minima", 0.4)

            if confiabilidad < confiabilidad_minima:
                self.logger.warning(f"Confiabilidad baja detectada: {confiabilidad:.2f}")
                self.config["umbral_confianza"] = min(self.config["umbral_confianza"] * 1.1, 0.9)

            return confiabilidad
        except Exception as e:
            self.logger.error(f"Error evaluando confiabilidad: {e}")
            return 1.0

    async def detectar_contradicciones(self):
        """Detecta contradicciones lógico-semánticas."""
        try:
            contradicciones = []
            for concepto, relaciones in self.memoria_semantica.items():
                for otro_concepto, peso in relaciones.items():
                    if peso > 0.5:
                        intuicion1 = self.intuiciones.get(concepto, 0.0)
                        intuicion2 = self.intuiciones.get(otro_concepto, 0.0)
                        if intuicion1 * intuicion2 < -0.25:
                            contradicciones.append({
                                "concepto1": concepto,
                                "concepto2": otro_concepto,
                                "intuicion1": intuicion1,
                                "intuicion2": intuicion2,
                                "timestamp": time.time()
                            })
            for c in contradicciones:
                afirmacion = (
                    f"Detecté una contradicción: {c['concepto1']} ({c['intuicion1']:.2f}) y "
                    f"{c['concepto2']} ({c['intuicion2']:.2f}) son opuestos."
                )
                if self.nucleus.db_pool:
                    async with self.nucleus.db_pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO cognitivo_contradicciones (instancia_id, contradiccion, timestamp)
                            VALUES ($1, $2, $3)
                            """,
                            self.nucleus.config.instance_id,
                            json.dumps(c),
                            c["timestamp"]
                        )
                await self.nucleus.publicar_alerta({
                    "tipo": "cognitivo_contradiccion",
                    "afirmacion": afirmacion,
                    "contradiccion": c,
                    "timestamp": c["timestamp"]
                })
                if c["intuicion1"] > c["intuicion2"]:
                    self.intuiciones[c["concepto2"]] *= 0.9
                else:
                    self.intuiciones[c["concepto1"]] *= 0.9
            self.logger.info(f"Contradicciones detectadas: {len(contradicciones)}")
            return contradicciones
        except Exception as e:
            self.logger.error(f"Error detectando contradicciones: {e}")
            return []

    async def detener(self):
        """Detiene el módulo cognitivo."""
        try:
            await self.guardar_estado()
            self.memoria.clear()
            self.intuiciones.clear()
            self.percepciones.clear()
            self.decisiones.clear()
            self.decisiones_fallidas.clear()
            self.contexto.clear()
            self.memoria_semantica.clear()
            self.intenciones.clear()
            self.atencion = {"focos": [], "nivel": 0.5}
            self.yo = {
                "estado": {"confianza": 1.0, "estabilidad": 1.0, "actividad": 0.0},
                "memoria": {"conceptos": {}, "tamaño": 0}
            }
            self.logger.info("Módulo Cognitivo detenido")
        except Exception:
            # Excepción capturada pero no usada, ya que el manejo de errores se realiza en el logger anterior
            pass
