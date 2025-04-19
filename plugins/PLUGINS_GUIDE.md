# Guía Completa para Crear Plugins en CoreC

Esta guía proporciona una explicación detallada y práctica para programadores que deseen desarrollar plugins para CoreC, un ecosistema digital bioinspirado diseñado para ser ligero, escalable, y modular. Los plugins son extensiones plug-and-play que actúan como "bloques de LEGO", integrándose dinámicamente con CoreC sin modificar su núcleo. Cada plugin puede añadir nuevas habilidades, como comunicación, razonamiento, o análisis, manteniendo la eficiencia (~1 KB por entidad, ~1 MB por bloque, ≤1 GB RAM para ~1M entidades) y escalabilidad (multi-nodo).

## Tabla de Contenidos

1. [Visión de los Plugins](#visión-de-los-plugins)
2. [Arquitectura de Plugins](#arquitectura-de-plugins)
3. [Estructura de un Plugin](#estructura-de-un-plugin)
4. [Interfaz del Plugin](#interfaz-del-plugin)
5. [Desarrollo Paso a Paso](#desarrollo-paso-a-paso)
6. [Mejores Prácticas](#mejores-prácticas)
7. [Ejemplo Completo: Plugin AnalizadorDatos](#ejemplo-completo-plugin-analizadordatos)
8. [Pruebas y Depuración](#pruebas-y-depuración)
9. [Despliegue en Producción](#despliegue-en-producción)
10. [Recursos Adicionales](#recursos-adicionales)

---

## Visión de los Plugins

En CoreC, los plugins son **extensiones neuronales** que amplían las capacidades del sistema, como si fueran células especializadas en un organismo vivo. Cada plugin:
- Es **independiente**: Funciona sin modificar el núcleo de CoreC, con su propio código, configuración, y dependencias.
- Es **sinérgico**: Colabora con otros plugins y bloques vía Redis streams y PostgreSQL, formando una red distribuida.
- Potencia el **razonamiento**: Añade habilidades como análisis, predicción, o comunicación, haciendo a CoreC más inteligente.
- Mantiene la **ligereza**: Usa entidades ultraligeras (~1 KB) y bloques simbióticos (~1 MB), respetando los límites de memoria.

Los plugins heredan la tecnología de CoreC (entidades, bloques, comunicación binaria, compresión `zstd`) para garantizar eficiencia y compatibilidad.

---

## Arquitectura de Plugins

Los plugins se integran con CoreC a través del núcleo (`corec/nucleus.py`), que actúa como el "cerebro" del sistema. La arquitectura incluye:

- **Cargador Dinámico**: `nucleus.py` escanea el directorio `plugins/` y carga plugins habilitados al iniciar CoreC.
- **Interfaz Estándar**: Los plugins implementan métodos `inicializar`, `ejecutar`, y `detener` para interactuar con el núcleo.
- **Comunicación**:
  - **Redis Streams**: Para enviar/recibir mensajes binarios entre plugins y bloques.
  - **PostgreSQL**: Para almacenar datos persistentes (por ejemplo, alertas, métricas).
- **Bloques Simbióticos**: Los plugins suelen crear sus propios bloques (`BloqueSimbiotico`) con entidades ultraligeras (`MicroCeluEntidadCoreC`) para procesar datos.
- **Autoreparación**: Los plugins pueden usar la lógica de autoreparación de CoreC (`IsolationForest`) para detectar y corregir anomalías.

---

## Estructura de un Plugin

Un plugin debe seguir esta estructura mínima:

plugins// ├── main.py # Lógica principal del plugin ├── config.json # Configuración específica (canales, parámetros) ├── requirements.txt # Dependencias específicas (opcional) ├── models/ # Modelos de IA o datos preentrenados (opcional) ├── data/ # Almacenamiento local para datos (opcional) ├── README.md # Documentación del plugin
- **`main.py`**: Contiene la clase principal del plugin y la función `inicializar(nucleus, config)`.
- **`config.json`**: Define parámetros como el canal Redis, número de entidades, y configuraciones específicas.
- **`requirements.txt`**: Lista dependencias adicionales (por ejemplo, `torch` para IA).
- **`models/`**: Almacena modelos preentrenados (por ejemplo, `.pt` para `torch`, `.pkl` para `scikit-learn`).
- **`data/`**: Guarda datos locales, como registros de entrenamiento.
- **`README.md`**: Explica el propósito, instalación, y uso del plugin.

---

## Interfaz del Plugin

Cada plugin debe implementar una interfaz estándar para integrarse con CoreC:

### **Función Obligatoria**
```python
def inicializar(nucleus, config):
    """
    Inicializa el plugin y lo registra en el núcleo.

    Args:
        nucleus: Instancia de CoreCNucleus (corec.nucleus.CoreCNucleus).
        config: Diccionario con la configuración del plugin (de config.json).
    """
    plugin = MiPlugin(nucleus, config)
    asyncio.create_task(plugin.inicializar())
Clase del Plugin
La clase principal debe implementar al menos:
class MiPlugin:
    def __init__(self, nucleus, config):
        self.nucleus = nucleus  # Acceso al núcleo
        self.config = config    # Configuración del plugin
        self.logger = logging.getLogger(f"Plugin-{config.get('nombre')}")

    async def inicializar(self):
        """Configura el plugin (por ejemplo, crea bloques, inicializa modelos)."""
        pass

    async def ejecutar(self):
        """Ejecuta la lógica principal del plugin (por ejemplo, procesa mensajes)."""
        pass

    async def detener(self):
        """Limpia recursos al detener el plugin."""
        pass
Registro en el Núcleo
El plugin se registra en CoreC con:
self.nucleus.registrar_plugin(self.config["nombre"], self)

Desarrollo Paso a Paso
1. Crear la Estructura del Plugin
mkdir -p plugins/mi_plugin/{models,data}
touch plugins/mi_plugin/main.py
touch plugins/mi_plugin/config.json
touch plugins/mi_plugin/requirements.txt
touch plugins/mi_plugin/README.md
2. Definir la Configuración (`config.json`)
Ejemplo:
{
    "nombre": "mi_plugin",
    "enabled": true,
    "canal": 6,
    "entidades": 100,
    "carga": 0.5,
    "intervalo": 60
}
* nombre: Identificador único del plugin.
* enabled: Habilita/desactiva el plugin (true/false).
* canal: Canal Redis para comunicación (por ejemplo, 6).
* entidades: Número de entidades ultraligeras (~1 KB cada una).
* carga: Factor de carga para el bloque simbiótico (0.0 a 1.0).
* intervalo: Frecuencia de ejecución en segundos.
3. Implementar la Lógica (`main.py`)
Ejemplo básico:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/mi_plugin/main.py
Plugin que añade una habilidad a CoreC.
"""
import asyncio
import logging
from corec.core import serializar_mensaje
from corec.entities import crear_entidad
from corec.blocks import BloqueSimbiotico
from typing import Dict, Any

class MiPlugin:
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.config = config
        self.logger = logging.getLogger(f"Plugin-{config.get('nombre')}")
        self.canal = config.get("canal", 6)
        self.bloque = None

    async def inicializar(self):
        entidades = [crear_entidad(f"m{i}", self.canal, self._procesar) for i in range(self.config.get("entidades", 100))]
        self.bloque = BloqueSimbiotico(f"{self.config['nombre']}", self.canal, entidades, nucleus=self.nucleus)
        self.nucleus.modulos["registro"].bloques[self.bloque.id] = self.bloque
        self.nucleus.registrar_plugin(self.config["nombre"], self)
        self.logger.info(f"Plugin {self.config['nombre']} inicializado")

    async def _procesar(self):
        return {"valor": random.random()}

    async def ejecutar(self):
        while True:
            resultado = await self.bloque.procesar(self.config.get("carga", 0.5))
            await self.nucleus.publicar_alerta({
                "tipo": f"resultado_{self.config['nombre']}",
                "bloque_id": self.bloque.id,
                "fitness": resultado["fitness"]
            })
            await asyncio.sleep(self.config.get("intervalo", 60))

    async def detener(self):
        self.logger.info(f"Plugin {self.config['nombre']} detenido")

def inicializar(nucleus, config):
    plugin = MiPlugin(nucleus, config)
    asyncio.create_task(plugin.inicializar())
4. Añadir Dependencias (`requirements.txt`)
Si el plugin usa bibliotecas adicionales:
numpy==1.24.3
Instalar:
cd plugins/mi_plugin
pip install -r requirements.txt
5. Crear Modelos o Datos (Opcional)
Si el plugin usa IA, inicializa modelos en models/:
import torch
torch.save(model.state_dict(), "plugins/mi_plugin/models/model.pt")
6. Documentar (`README.md`)
Incluye propósito, instalación, uso, y ejemplos. Ver el README.md de comunicador_inteligente como referencia.
7. Probar el Plugin
Crear pruebas en tests/test_mi_plugin.py:
#!/usr/bin/env python3
import unittest
import asyncio
from unittest.mock import AsyncMock
from plugins.mi_plugin.main import MiPlugin

class TestMiPlugin(unittest.TestCase):
    def setUp(self):
        self.config = {"nombre": "mi_plugin", "canal": 6, "entidades": 10}
        self.nucleus = AsyncMock()
        self.nucleus.modulos = {"registro": AsyncMock()}
        self.plugin = MiPlugin(self.nucleus, self.config)
        self.loop = asyncio.get_event_loop()

    async def test_inicializar(self):
        await self.plugin.inicializar()
        self.assertIsNotNone(self.plugin.bloque)
        self.nucleus.registrar_plugin.assert_called()

    def test_all(self):
        self.loop.run_until_complete(self.test_inicializar())

if __name__ == "__main__":
    unittest.main()
Ejecutar:
python -m unittest tests/test_mi_plugin.py -v
8. Integrar con CoreC
* Coloca el plugin en plugins/mi_plugin/.
* Ejecuta CoreC: bash run.sh
* 

Mejores Prácticas
1. Ligereza:
    * Limita entidades a ~100-1000 por bloque (~100 KB a ~1 MB).
    * Usa compresión zstd para datos y modelos.
    * Evita dependencias pesadas; prioriza bibliotecas ligeras como scikit-learn.
2. Modularidad:
    * Haz el plugin autocontenido, con dependencias en requirements.txt.
    * Usa config.json para parámetros configurables.
3. Resiliencia:
    * Implementa manejo de errores en inicializar, ejecutar, y detener.
    * Usa la autoreparación de BloqueSimbiotico para gestionar fallos.
4. Comunicación:
    * Usa Redis streams para mensajes entre plugins y bloques.
    * Serializa mensajes con serializar_mensaje para eficiencia.
5. Pruebas:
    * Crea pruebas unitarias e integrales en tests/.
    * Simula fallos (Redis desconectado, datos corruptos) con unittest.mock.
6. Documentación:
    * Incluye un README.md claro.
    * Documenta la interfaz y configuración en config.json.
7. Escalabilidad:
    * Diseña plugins para ejecutarse en nodos separados, usando Redis para coordinación.
    * Minimiza escrituras en PostgreSQL, usando compresión zstd.

Ejemplo Completo: Plugin AnalizadorDatos
Este plugin analiza datos de otros bloques y predice anomalías usando una red neuronal ligera.
Estructura
plugins/analizador_datos/
├── main.py
├── config.json
├── requirements.txt
├── models/
│   └── nn.pt
├── data/
│   └── training.log
├── README.md
`main.py`
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/analizador_datos/main.py
Plugin que analiza datos de bloques y predice anomalías.
"""
import asyncio
import logging
import json
import torch
import torch.nn as nn
from corec.core import serializar_mensaje, deserializar_mensaje, aioredis
from corec.entities import crear_entidad
from corec.blocks import BloqueSimbiotico
from typing import Dict, Any

class RedNeuronalAnalizadora(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class AnalizadorDatos:
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.config = config
        self.logger = logging.getLogger(f"Plugin-{config.get('nombre')}")
        self.canal = config.get("canal", 7)
        self.bloque = None
        self.nn_model = RedNeuronalAnalizadora()
        self.redis_client = None
        self.model_path = "plugins/analizador_datos/models/nn.pt"
        self.training_log = "plugins/analizador_datos/data/training.log"

    async def inicializar(self):
        # Inicializar Redis
        redis_url = f"redis://{self.nucleus.redis_config['username']}:{self.nucleus.redis_config['password']}@{self.nucleus.redis_config['host']}:{self.nucleus.redis_config['port']}"
        self.redis_client = await aioredis.from_url(redis_url, decode_responses=True)
        self.logger.info("Redis inicializado")

        # Cargar modelo
        try:
            with open(self.model_path, "rb") as f:
                self.nn_model.load_state_dict(torch.load(f))
        except FileNotFoundError:
            torch.save(self.nn_model.state_dict(), self.model_path)

        # Crear bloque simbiótico
        entidades = [crear_entidad(f"m{i}", self.canal, self._analizar) for i in range(self.config.get("entidades", 100))]
        self.bloque = BloqueSimbiotico(f"analizador_datos", self.canal, entidades, nucleus=self.nucleus)
        self.nucleus.modulos["registro"].bloques[self.bloque.id] = self.bloque
        self.nucleus.registrar_plugin(self.config["nombre"], self)
        self.logger.info(f"Plugin AnalizadorDatos inicializado")

        # Iniciar escucha de datos
        asyncio.create_task(self._escuchar_datos())

    async def _escuchar_datos(self):
        stream = "bloque_data_stream"
        while True:
            try:
                mensajes = await self.redis_client.xread({stream: "0-0"}, count=10)
                for _, entries in mensajes:
                    for _, data in entries:
                        mensaje = await deserializar_mensaje(data["data"])
                        resultado = await self._analizar(mensaje)
                        await self.nucleus.publicar_alerta({
                            "tipo": "analisis_datos",
                            "bloque_id": self.bloque.id,
                            "prediccion": resultado["valor"]
                        })
            except Exception as e:
                self.logger.error(f"Error escuchando datos: {e}")
            await asyncio.sleep(1)

    async def _analizar(self, mensaje: Dict[str, Any] = None):
        valor = mensaje.get("valor", random.random()) if mensaje else random.random()
        fitness = mensaje.get("fitness", 0.5) if mensaje else 0.5
        input_tensor = torch.tensor([valor, fitness], dtype=torch.float32)
        with torch.no_grad():
            prediccion = self.nn_model(input_tensor).item()
        with open(self.training_log, "a") as f:
            f.write(json.dumps({"entrada": [valor, fitness], "salida": prediccion}) + "\n")
        return {"valor": prediccion}

    async def ejecutar(self):
        while True:
            resultado = await self.bloque.procesar(self.config.get("carga", 0.5))
            await self.nucleus.publicar_alerta({
                "tipo": "analisis_datos",
                "bloque_id": self.bloque.id,
                "fitness": resultado["fitness"]
            })
            await asyncio.sleep(self.config.get("intervalo", 60))

    async def detener(self):
        self.logger.info("Plugin AnalizadorDatos detenido")
        if self.redis_client:
            await self.redis_client.close()
        torch.save(self.nn_model.state_dict(), self.model_path)

def inicializar(nucleus, config):
    plugin = AnalizadorDatos(nucleus, config)
    asyncio.create_task(plugin.inicializar())
`config.json`
{
    "nombre": "analizador_datos",
    "enabled": true,
    "canal": 7,
    "entidades": 100,
    "carga": 0.5,
    "intervalo": 60
}
`requirements.txt`
torch==2.0.1
`README.md`
# Plugin AnalizadorDatos para CoreC

Analiza datos de bloques de CoreC y predice anomalías usando una red neuronal ligera.

## Instalación

1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
1. Inicializa modelo: python -c "from main import RedNeuronalAnalizadora; import torch; nn=RedNeuronalAnalizadora(); torch.save(nn.state_dict(), 'models/nn.pt')"
2. touch data/training.log
3. 
Uso
* Los datos de otros bloques se leen desde bloque_data_stream.
* Las predicciones se publican como alertas en CoreC.
### **`models/nn.pt`**
```python
import torch
from plugins.analizador_datos.main import RedNeuronalAnalizadora
nn = RedNeuronalAnalizadora()
torch.save(nn.state_dict(), "plugins/analizador_datos/models/nn.pt")
`data/training.log`
mkdir -p plugins/analizador_datos/data
touch plugins/analizador_datos/data/training.log

Pruebas y Depuración
1. Crear Pruebas:
    * Usa unittest para probar inicializar, ejecutar, y lógica específica.
    * Simula Redis y PostgreSQL con unittest.mock.
2. Depuración:
    * Habilita logging detallado en main.py: logging.basicConfig(level=logging.DEBUG)
    * 
    * Monitorea streams Redis con: redis-cli MONITOR
    * 
3. Validar Memoria:
    * Usa tracemalloc para medir consumo: import tracemalloc
    * tracemalloc.start()
    * # Código del plugin
    * snapshot = tracemalloc.take_snapshot()
    * print(snapshot.statistics("lineno"))
    * 

Despliegue en Producción
1. Configuración:
    * Usa variables de entorno para credenciales: export OPENROUTER_API_KEY="tu_api_key"
    * 
    * Actualiza config.json para producción: {
    *     "openrouter_api_key": "${OPENROUTER_API_KEY}"
    * }
    * 
2. Docker:
    * Añade el plugin al Dockerfile: COPY plugins/comunicador_inteligente /app/plugins/comunicador_inteligente
    * RUN pip install -r /app/plugins/comunicador_inteligente/requirements.txt
    * 
3. Monitoreo:
    * Configura métricas en monitoring/prometheus.yml: - job_name: 'plugin_analizador_datos'
    *   static_configs:
    *     - targets: ['corec:8000']
    * 
4. Escalabilidad:
    * Ejecuta plugins en nodos separados, configurando instance_id en corec_config.json.
    * Usa Redis para coordinar comunicación.

Recursos Adicionales
* Documentación de CoreC: README.md en el directorio raíz.
* Ejemplo Avanzado: Estudia plugins/comunicador_inteligente/.
* API de OpenRouter: https://openrouter.ai/docs
* Bibliotecas Recomendadas:
    * torch: Para redes neuronales ligeras.
    * scikit-learn: Para modelos ML simples.
    * aiohttp: Para APIs asíncronas.

Esta guía te permite crear plugins robustos y escalables para CoreC. ¡Construye habilidades nuevas y haz que CoreC evolucione como un organismo vivo!
---
