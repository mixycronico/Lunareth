# corec/config.py
"""
Configuración global de CoreC.
Define constantes y configuraciones predeterminadas para el sistema.
"""
QUANTIZATION_STEP_DEFAULT = 0.05  # Paso de cuantización por defecto
MAX_ENLACES_POR_ENTIDAD = 100     # Máximo número de enlaces por entidad
REDIS_STREAM_KEY = "corec:entrelazador"  # Clave para persistencia de enlaces en Redis
ALERT_THRESHOLD = 0.9             # Umbral para alertas de recursos (90% CPU/memoria)
MAX_FALLOS_CRITICOS = 0.5         # Porcentaje de fallos para considerar un bloque crítico
CPU_AUTOADJUST_THRESHOLD = 0.9    # Umbral de CPU para autoajuste (90%)
RAM_AUTOADJUST_THRESHOLD = 0.9    # Umbral de RAM para autoajuste (90%)
CONCURRENT_TASKS_MIN = 10         # Mínimo número de tareas concurrentes
CONCURRENT_TASKS_MAX = 1000       # Máximo número de tareas concurrentes
CONCURRENT_TASKS_REDUCTION_FACTOR = 0.8  # Factor de reducción cuando CPU/RAM > 90%
CONCURRENT_TASKS_INCREMENT_FACTOR_DEFAULT = 1.05  # Factor de incremento inicial
CPU_STABLE_CYCLES = 3             # Ciclos consecutivos de CPU baja para incrementar tareas
CPU_READINGS = 3                  # Número de lecturas para promediar CPU
CPU_READING_INTERVAL = 0.05       # Intervalo entre lecturas de CPU (segundos)
PERFORMANCE_HISTORY_SIZE = 10     # Tamaño del historial para aprendizaje de rendimiento
PERFORMANCE_THRESHOLD = 0.5       # Umbral de tiempo de procesamiento (segundos) para ajustar incremento
INCREMENT_FACTOR_MIN = 1.01       # Mínimo factor de incremento
INCREMENT_FACTOR_MAX = 1.1        # Máximo factor de incremento
