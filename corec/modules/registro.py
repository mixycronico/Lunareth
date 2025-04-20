import logging
import random
from typing import Dict
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad


class ModuloRegistro(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloRegistro")
        self.nucleus = None
        self.bloques: Dict[str, Dict] = {}

    async def inicializar(self, nucleus, config=None):
        """Inicializa el módulo de registro."""
        try:
            self.nucleus = nucleus
            self.logger.info("[Registro] Módulo inicializado")
        except Exception as e:
            self.logger.error(f"[Registro] Error inesperado al inicializar: {e}")

    async def registrar_bloque(self, bloque_id: str, canal: int, num_entidades: int, max_size_mb: float = 10.0):
        """Registra un bloque simbiótico."""
        try:
            if not bloque_id or canal < 0 or num_entidades <= 0:
                raise ValueError("Configuración inválida para el bloque")
            async def test_func(): return {"valor": 0.7}
            entidades = [crear_entidad(f"m{i}", canal, test_func) for i in range(num_entidades)]
            BloqueSimbiotico(bloque_id, canal, entidades, max_size_mb, self.nucleus)
            self.bloques[bloque_id] = {
                "canal": canal,
                "num_entidades": num_entidades,
                "fitness": 0.0,
                "timestamp": random.random()
            }
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_registrado",
                "bloque_id": bloque_id,
                "num_entidades": num_entidades,
                "timestamp": random.random()
            })
            self.logger.info(f"[Registro] Bloque '{bloque_id}' registrado")
        except Exception as e:
            self.logger.error(f"[Registro] Error registrando bloque '{bloque_id}': {e}")
            raise

    async def detener(self):
        """Detiene el módulo de registro."""
        self.logger.info("[Registro] Módulo detenido")
