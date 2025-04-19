from corec.core import ModuloBase, celery_app, logging, asyncio, time, random
from corec.entities import MicroCeluEntidadCoreC, CeluEntidadCoreC, crear_entidad, crear_celu_entidad, procesar_entidad, procesar_celu_entidad
from corec.blocks import BloqueSimbiotico

class ModuloRegistro(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloRegistro")
        self.bloques = {}

    async def inicializar(self, nucleus):
        self.nucleus = nucleus
        self.logger.info("[ModuloRegistro] Inicializado")
        for bloque_config in self.nucleus.config.get("bloques", []):
            await self.registrar_bloque(bloque_config["id"], bloque_config["canal"], bloque_config["entidades"])

    async def registrar_bloque(self, bloque_id: str, canal: int, num_entidades: int):
        entidades_por_bloque = 1000
        num_bloques = (num_entidades // entidades_por_bloque) + (1 if num_entidades % entidades_por_bloque else 0)
        for j in range(num_bloques):
            bloque_id_j = f"{bloque_id}_{j}" if j else bloque_id
            entidades = []
            for i in range(min(entidades_por_bloque, num_entidades - j * entidades_por_bloque)):
                async def funcion():
                    return {"valor": random.random()}
                entidades.append(crear_entidad(f"m{i+j*entidades_por_bloque}", canal, funcion))
            bloque = BloqueSimbiotico(bloque_id_j, canal, entidades, nucleus=self.nucleus)
            self.bloques[bloque_id_j] = bloque
            self.logger.info(f"Bloque {bloque_id_j} registrado con {len(entidades)} entidades")

    async def registrar_celu_entidad(self, celu: CeluEntidadCoreC):
        bloque_id = f"celu_{celu.id}"
        bloque = BloqueSimbiotico(bloque_id, celu.canal, [], nucleus=self.nucleus)
        self.bloques[bloque_id] = bloque
        self.logger.info(f"CeluEntidad {celu.id} registrada en bloque {bloque_id}")

    async def registrar_micro_celu_entidad(self, micro: MicroCeluEntidadCoreC):
        bloque_id = f"micro_{micro.id}"
        bloque = BloqueSimbiotico(bloque_id, micro.canal, [micro], nucleus=self.nucleus)
        self.bloques[bloque_id] = bloque
        self.logger.info(f"MicroCeluEntidad {micro.id} registrada en bloque {bloque_id}")

    async def ejecutar(self):
        while True:
            for bloque in self.bloques.values():
                await bloque.escribir_postgresql(self.nucleus.db_config)
            await asyncio.sleep(300)

    async def detener(self):
        self.logger.info("[ModuloRegistro] Detenido")