import time
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from corec.entities_superpuestas import EntidadSuperpuesta


class ModuloRegistro(ComponenteBase):
    def __init__(self):
        self.nucleus = None
        self.bloques = {}

    async def inicializar(self, nucleus, config=None):
        """Inicializa el módulo de registro.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del módulo (opcional).
        """
        try:
            self.nucleus = nucleus
            self.logger = nucleus.logger
            self.logger.info("Módulo Registro inicializado")
        except Exception as e:
            self.logger.error(f"Error inicializando Módulo Registro: {e}")
            raise

    async def registrar_bloque(self, bloque_id: str, canal: int, num_entidades: int, max_size_mb: float = 10.0):
        """Registra un bloque simbiótico.

        Args:
            bloque_id (str): Identificador único del bloque.
            canal (int): Canal de comunicación.
            num_entidades (int): Número de entidades en el bloque.
            max_size_mb (float): Tamaño máximo en MB.

        Raises:
            ValueError: Si la configuración del bloque es inválida.
        """
        try:
            if not bloque_id or canal < 0 or num_entidades <= 0:
                raise ValueError("Configuración inválida para el bloque")

            entidades = [
                EntidadSuperpuesta(
                    f"{bloque_id}_ent_{i}",
                    {"rol1": 0.5, "rol2": 0.5},
                    quantization_step=self.nucleus.config.quantization_step_default,
                    min_fitness=0.3,
                    mutation_rate=0.1
                )
                for i in range(num_entidades)
            ]
            bloque = BloqueSimbiotico(
                bloque_id,
                canal,
                entidades,
                max_size_mb,
                self.nucleus,
                quantization_step=self.nucleus.config.quantization_step_default,
                max_errores=self.nucleus.config.max_fallos_criticos,
                max_concurrent_tasks=self.nucleus.config.concurrent_tasks_max,
                cpu_intensive=False
            )
            self.bloques[bloque_id] = {
                "canal": canal,
                "num_entidades": num_entidades,
                "fitness": 0.0,
                "timestamp": time.time()
            }
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_registrado",
                "bloque_id": bloque_id,
                "num_entidades": num_entidades,
                "timestamp": time.time()
            })
            self.logger.info(f"Bloque '{bloque_id}' registrado con {num_entidades} entidades")
        except Exception as e:
            self.logger.error(f"Error registrando bloque '{bloque_id}': {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_registro",
                "bloque_id": bloque_id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def detener(self):
        """Detiene el módulo de registro."""
        self.logger.info("Módulo Registro detenido")
