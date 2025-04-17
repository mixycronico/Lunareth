Guía para Desarrolladores: Creación de Plugins para CoreC
CoreC es un sistema distribuido biomimético diseñado para ser extensible mediante plugins, permitiendo añadir funcionalidades sin modificar el núcleo (CoreCNucleus). En la Etapa 1, CoreC incluye una infraestructura de plugins vacía gestionada por PluginManager, lista para que los desarrolladores creen plugins que procesen datos en canales específicos, integren nuevas capacidades o interactúen con el sistema de manera conversacional. Esta guía detalla cómo crear plugins, desde la estructura y configuración hasta la implementación, pruebas y mejores prácticas, asegurando que se integren armónicamente con el núcleo.

Tabla de Contenidos
  1	Introducción
  2	Arquitectura de Plugins
  3	Requisitos Previos
  4	Estructura de un Plugin
  5	Pasos para Crear un Plugin
  6	Ejemplo Práctico: Plugin de Prueba
  7	Uso de OpenRouter en Plugins
  8	Pruebas de Plugins
  9	Mejores Prácticas
  10	Solución de Problemas
  11	Próximos Pasos

Introducción
Los plugins en CoreC son módulos independientes que extienden las funcionalidades del sistema, como procesar datos en canales específicos, implementar interfaces de usuario, o integrar servicios externos. Cada plugin se conecta al núcleo mediante PluginManager, que carga dinámicamente los plugins desde el directorio src/plugins/ según su configuración en plugin.json. Los plugins pueden aprovechar OpenRouter para análisis avanzado o chat, pero deben ser autónomos, siguiendo la filosofía de resiliencia del núcleo (16/04/2025, 22:07).
Esta guía está dirigida a desarrolladores que desean crear plugins para CoreC, ya sea para añadir capacidades específicas (como alertas o trading, 15/04/2025, 10:47, 14:21) o para experimentar con nuevas funcionalidades. Proporciona instrucciones detalladas, ejemplos de código y consejos prácticos.

Arquitectura de Plugins
Los plugins en CoreC están diseñados para ser modulares y biomiméticos, integrándose como “órganos” que colaboran con el núcleo (CoreCNucleus). La arquitectura incluye:
  •	PluginManager (src/plugins/plugin_manager.py):
  ◦	Escanea src/plugins/ para detectar plugins.
  ◦	Carga clases principales definidas en plugin.json.
  ◦	Registra canales asociados a cada plugin.
  ◦	Gestiona el ciclo de vida de los plugins (inicializar, detener).
  •	PluginBase (src/plugins/plugin_manager.py):
  ◦	Clase base para plugins, con métodos inicializar y detener.
  ◦	Proporciona acceso a redis_client, db_config, y el núcleo.
  •	ProcesadorBase (src/core/processors/base.py):
  ◦	Interfaz para plugins de tipo “procesador”, que procesan datos en canales específicos.
  ◦	Método principal: procesar(datos, contexto).
  •	Canales:
  ◦	Cada plugin define canales (ej., test_canal, alertas) que son procesados por CeluEntidadCoreC.
  ◦	Los canales críticos (en canales_criticos) generan espejos para resiliencia.
  •	OpenRouter:
  ◦	Plugins pueden usar nucleus.razonar y nucleus.responder_chat para análisis o chat, con fallbacks locales si no está disponible.
  •	Base de Datos:
  ◦	Plugins pueden usar la base de datos principal (corec_db) o crear tablas propias (ej., predictions para trading).
El flujo típico de un plugin es:
  1	PluginManager detecta el plugin en src/plugins//plugin.json.
  2	Carga la clase principal (que extiende PluginBase o ProcesadorBase).
  3	El plugin se inicializa, registrando sus canales.
  4	CeluEntidadCoreC usa el procesador del plugin para manejar datos en sus canales.
  5	El plugin puede interactuar con OpenRouter, PostgreSQL, Redis, o el núcleo según sea necesario.

Requisitos Previos
  •	CoreC Configurado: El núcleo de CoreC (Etapa 1) debe estar instalado y funcionando. Consulta la documentación principal para configurarlo.
  •	Python: 3.11+.
  •	Dependencias:
  ◦	Las del núcleo (requirements.txt).
  ◦	Dependencias específicas del plugin (definidas en plugin.json).
  •	Conocimientos:
  ◦	Python asíncrono (asyncio).
  ◦	PostgreSQL y Redis (opcional, según el plugin).
  ◦	OpenRouter API (opcional, para análisis o chat).
  •	Herramientas:
  ◦	Editor de código (VS Code, PyCharm).
  ◦	Docker (para pruebas locales).
  ◦	pytest para pruebas.

Estructura de un Plugin
Un plugin típico tiene la siguiente estructura:
src/plugins//
├── __init__.py
├── plugin.json               # Configuración del plugin
├── processors/
│   ├── __init__.py
│   ├── .py # Clase procesadora
├── utils/                    # Utilidades opcionales
│   ├── __init__.py
│   ├── .py
configs/plugins//
├── .yaml      # Configuración específica
  •	plugin.json:
  ◦	Define el nombre, versión, tipo, canales, dependencias y clase principal.
  ◦	Ejemplo: {
  ◦	  "name": "",
  ◦	  "version": "1.0.0",
  ◦	  "description": "Descripción del plugin",
  ◦	  "type": "processor",
  ◦	  "channels": ["canal1", "canal2"],
  ◦	  "dependencies": ["dependencia==versión"],
  ◦	  "config_file": "configs/plugins//.yaml",
  ◦	  "main_class": ".processors..",
  ◦	  "critical": false
  ◦	}
  ◦	
  •	.py:
  ◦	Contiene la clase que extiende ProcesadorBase o PluginBase.
  •	.yaml:
  ◦	Configuración específica del plugin (ej., endpoints, claves API).
  •	utils/:
  ◦	Funcionalidades auxiliares (ej., conexiones a bases de datos externas).

Pasos para Crear un Plugin
Paso 1: Crea el Directorio del Plugin
  1	Crea un directorio en src/plugins//: mkdir -p src/plugins//processors
  2	mkdir -p src/plugins//utils
  3	touch src/plugins//__init__.py
  4	touch src/plugins//processors/__init__.py
  5	touch src/plugins//utils/__init__.py
  6	
  7	Crea el directorio de configuración: mkdir -p configs/plugins/
  8	
Paso 2: Define plugin.json
Crea src/plugins//plugin.json con la configuración del plugin. Por ejemplo:
{
  "name": "mi_plugin",
  "version": "1.0.0",
  "description": "Plugin de prueba para CoreC",
  "type": "processor",
  "channels": ["mi_canal"],
  "dependencies": [],
  "config_file": "configs/plugins/mi_plugin/mi_plugin.yaml",
  "main_class": "mi_plugin.processors.mi_processor.MiProcessor",
  "critical": false
}
  •	name: Nombre único del plugin.
  •	version: Versión del plugin (formato semántico).
  •	description: Breve descripción.
  •	type: Tipo de plugin (processor para procesadores, interface para UI, etc.).
  •	channels: Canales que el plugin procesará.
  •	dependencies: Librerías requeridas (instaladas con pip).
  •	config_file: Ruta al archivo YAML de configuración.
  •	main_class: Ruta a la clase principal (relative al directorio src/plugins/).
  •	critical: Si true, los canales generan espejos para resiliencia.
Paso 3: Crea la Configuración
Crea configs/plugins//.yaml. Ejemplo:
channels:
  - "mi_canal"
# Configuraciones específicas
endpoint: "https://example.com/api"
Asegúrate de que el archivo sea accesible desde el contenedor Docker si usas docker-compose.yml.
Paso 4: Implementa la Clase Procesadora
Crea src/plugins//processors/.py con una clase que extienda ProcesadorBase. Ejemplo:
from ....core.processors.base import ProcesadorBase
from ....utils.logging import logger
from typing import Dict, Any

class MiProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("MiProcessor")

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.logger.info("MiProcessor inicializado")

    async def procesar(self, datos: Any, contexto: Dict[str, Any]) -> Any:
        # Procesar datos del canal
        resultado = {"procesado": datos.get("valores", [])}

        # Ejemplo: Usar OpenRouter para análisis
        analisis = await self.nucleus.razonar(resultado, f"Análisis para canal {contexto['canal']}")
        resultado["analisis"] = analisis["respuesta"]

        return {
            "estado": "ok",
            "resultado": resultado,
            "timestamp": contexto["timestamp"]
        }

    async def detener(self):
        self.logger.info("MiProcessor detenido")
  •	init: Inicializa el plugin con la configuración, Redis y la base de datos.
  •	inicializar: Configura el plugin, accediendo al núcleo.
  •	procesar: Procesa datos del canal, retornando un diccionario con estado y resultados.
  •	detener: Libera recursos (ej., conexiones).
Paso 5: Instala Dependencias (si las hay)
Si el plugin tiene dependencias, añádelas a plugin.json:
"dependencies": ["requests==2.31.0"]
Instala las dependencias:
pip install requests==2.31.0
O usa un script para automatizar:
#!/bin/bash
PLUGIN_NAME=$1
jq -r '.dependencies[]' "src/plugins/$PLUGIN_NAME/plugin.json" | xargs pip install
Paso 6: Prueba el Plugin
  1	Inicia CoreC: ./scripts/start.sh
  2	
  3	Simula datos en el canal: docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('mi_canal', '{\"valores\": [1, 2, 3]}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  4	
  5	Revisa los logs: docker logs corec_v4-corec1-1
  6	 Busca mensajes como [MiProcessor] Procesado.
Paso 7: Escribe Pruebas
Crea pruebas en tests/plugins/test_.py. Ejemplo:
# tests/plugins/test_mi_plugin.py
import pytest
import asyncio
from src.plugins.mi_plugin.processors.mi_processor import MiProcessor
from src.utils.config import load_secrets

@pytest.mark.asyncio
async def test_mi_processor():
    config = load_secrets("configs/plugins/mi_plugin/mi_plugin.yaml")
    processor = MiProcessor(config, None, None)
    await processor.inicializar(None)
    result = await processor.procesar({"valores": [1, 2, 3]}, {"timestamp": 1234567890, "canal": "mi_canal"})
    assert result["estado"] == "ok"
    assert "procesado" in result["resultado"]
    await processor.detener()
Ejecuta las pruebas:
pytest tests/plugins/test_mi_plugin.py

Ejemplo Práctico: Plugin de Prueba
Creemos un plugin simple que procese datos en el canal prueba_canal.
Estructura
src/plugins/prueba/
├── __init__.py
├── plugin.json
├── processors/
│   ├── __init__.py
│   ├── prueba_processor.py
configs/plugins/prueba/
├── prueba.yaml
tests/plugins/
├── test_prueba.py
plugin.json
{
  "name": "prueba",
  "version": "1.0.0",
  "description": "Plugin de prueba para CoreC",
  "type": "processor",
  "channels": ["prueba_canal"],
  "dependencies": [],
  "config_file": "configs/plugins/prueba/prueba.yaml",
  "main_class": "prueba.processors.prueba_processor.PruebaProcessor",
  "critical": false
}
prueba.yaml
channels:
  - "prueba_canal"
procesador:
  max_datos: 100
prueba_processor.py
from ....core.processors.base import ProcesadorBase
from ....utils.logging import logger
from typing import Dict, Any

class PruebaProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("PruebaProcessor")
        self.max_datos = self.config.get("procesador", {}).get("max_datos", 100)

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.logger.info("PruebaProcessor inicializado")

    async def procesar(self, datos: Any, contexto: Dict[str, Any]) -> Any:
        valores = datos.get("valores", [])[:self.max_datos]
        if not valores:
            return {"estado": "error", "mensaje": "No hay datos para procesar"}

        # Análisis simple
        promedio = sum(valores) / len(valores) if valores else 0

        # Usar OpenRouter para análisis adicional
        analisis = await self.nucleus.razonar({"valores": valores}, f"Análisis de datos en {contexto['canal']}")
        resultado_analisis = analisis["respuesta"]

        return {
            "estado": "ok",
            "promedio": promedio,
            "analisis": resultado_analisis,
            "timestamp": contexto["timestamp"]
        }

    async def detener(self):
        self.logger.info("PruebaProcessor detenido")
test_prueba.py
import pytest
import asyncio
from src.plugins.prueba.processors.prueba_processor import PruebaProcessor
from src.utils.config import load_secrets

@pytest.mark.asyncio
async def test_prueba_processor(monkeypatch):
    async def mock_razonar(self, datos, contexto):
        return {"estado": "ok", "respuesta": "Análisis local: datos recibidos"}

    monkeypatch.setattr("src.core.nucleus.CoreCNucleus.razonar", mock_razonar)

    config = load_secrets("configs/plugins/prueba/prueba.yaml")
    processor = PruebaProcessor(config, None, None)
    await processor.inicializar(None)
    result = await processor.procesar({"valores": [1, 2, 3]}, {"timestamp": 1234567890, "canal": "prueba_canal"})
    assert result["estado"] == "ok"
    assert result["promedio"] == 2.0
    assert "analisis" in result
    await processor.detener()
Configuración y Prueba
  1	Crea los directorios y archivos.
  2	Actualiza main.py para registrar una entidad en prueba_canal: await nucleus.registrar_celu_entidad(
  3	    CeluEntidadCoreC(
  4	        f"nano_prueba_{instance_id}",
  5	        nucleus.get_procesador("prueba_canal"),
  6	        "prueba_canal",
  7	        5.0,
  8	        nucleus.db_config,
  9	        instance_id=instance_id
  10	    )
  11	)
  12	
  13	Inicia CoreC: ./scripts/start.sh
  14	
  15	Simula datos: docker exec corec_v4-postgres-1 psql -U corec_user -d corec_db -c "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES ('prueba_canal', '{\"valores\": [10, 20, 30]}', EXTRACT(EPOCH FROM NOW()), 'corec1');"
  16	
  17	Revisa los logs: docker logs corec_v4-corec1-1
  18	
  19	Ejecuta la prueba: pytest tests/plugins/test_prueba.py
  20	

Uso de OpenRouter en Plugins
Los plugins pueden usar OpenRouter para análisis avanzado o chat, accediendo a los métodos del núcleo:
  •	Análisis: analisis = await self.nucleus.razonar(datos, "Contexto del análisis")
  •	resultado = analisis["respuesta"] if analisis["estado"] == "ok" else "Análisis local"
  •	
  •	Chat: respuesta = await self.nucleus.responder_chat("Mensaje del usuario", "Contexto")
  •	mensaje = respuesta["respuesta"]
  •	
Consideraciones:
  •	Fallbacks: Si OpenRouter no está disponible (enabled: false o sin conexión), el núcleo proporciona respuestas locales (definidas en openrouter.py).
  •	Límites: Configura max_tokens y temperature en openrouter.yaml para controlar el uso.
  •	Caching: Considera almacenar respuestas frecuentes en Redis (Etapa 2).
  •	Seguridad: Valida los datos enviados a OpenRouter para evitar inyecciones.

Pruebas de Plugins
Cada plugin debe incluir pruebas unitarias en tests/plugins/test_.py. Sigue estas prácticas:
  •	Prueba la inicialización, procesamiento y detención del plugin.
  •	Simula datos de entrada para el canal.
  •	Usa monkeypatch para simular OpenRouter o dependencias externas.
  •	Verifica que los fallbacks funcionen correctamente.
Ejemplo:
@pytest.mark.asyncio
async def test_mi_processor_fallback(monkeypatch):
    async def mock_razonar(self, datos, contexto):
        return {"estado": "fallback", "respuesta": "No se pudo conectar"}

    monkeypatch.setattr("src.core.nucleus.CoreCNucleus.razonar", mock_razonar)

    config = load_secrets("configs/plugins/mi_plugin/mi_plugin.yaml")
    processor = MiProcessor(config, None, None)
    await processor.inicializar(None)
    result = await processor.procesar({"valores": [1, 2, 3]}, {"timestamp": 1234567890, "canal": "mi_canal"})
    assert result["estado"] == "ok"
    assert result["analisis"] == "No se pudo conectar"

Mejores Prácticas
  •	Modularidad:
  ◦	Mantén el plugin independiente, usando solo las interfaces proporcionadas (ProcesadorBase, PluginBase).
  ◦	Evita dependencias directas con otros plugins.
  •	Configuración:
  ◦	Define configuraciones en .yaml, no en el código.
  ◦	Usa config para parámetros ajustables.
  •	Resiliencia:
  ◦	Maneja excepciones en procesar y inicializar.
  ◦	Implementa fallbacks para OpenRouter y servicios externos.
  •	Pruebas:
  ◦	Cubre todos los métodos públicos del plugin.
  ◦	Simula fallos de red o dependencias.
  •	Logging:
  ◦	Usa self.logger para registrar eventos y errores.
  ◦	Ejemplo: self.logger.info("Procesado exitoso").
  •	Escalabilidad:
  ◦	Limita el tamaño de los datos procesados (ej., max_datos).
  ◦	Usa Redis para datos temporales en lugar de la base de datos.
  •	Seguridad:
  ◦	Valida datos de entrada para evitar inyecciones.
  ◦	No almacenes credenciales en el código.

Solución de Problemas
  •	Plugin no se carga:
  ◦	Verifica que plugin.json sea válido y main_class sea correcto.
  ◦	Revisa los logs: docker logs corec_v4-corec1-1.
  ◦	Asegúrate de que configs/plugins//.yaml exista.
  •	Canal no procesado:
  ◦	Confirma que el canal está en plugin.json (channels).
  ◦	Registra una CeluEntidadCoreC para el canal en main.py.
  •	Error de dependencias:
  ◦	Instala las dependencias listadas en plugin.json: pip install 
  ◦	
  ◦	Verifica la compatibilidad con requirements.txt.
  •	OpenRouter falla:
  ◦	Asegúrate de que enabled: true y la clave API sean correctos en openrouter.yaml.
  ◦	Prueba con enabled: false para usar fallbacks.
  •	Pruebas fallan:
  ◦	Revisa los logs de pytest: pytest tests/plugins/test_.py -v.
  ◦	Confirma que la configuración del plugin sea accesible.

Próximos Pasos
  •	Implementar Plugins Específicos:
  ◦	CLI: Interfaz interactiva con Textual.
  ◦	Alertas: Notificaciones externas para el canal alertas.
  ◦	Trading: Análisis de mercado con redes neuronales y base de datos propia.
  •	Optimizar OpenRouter:
  ◦	Añadir caching en Redis para respuestas frecuentes.
  ◦	Configurar límites de uso por plugin.
  •	Escalabilidad:
  ◦	Probar plugins con múltiples instancias en Kubernetes.
  ◦	Optimizar el procesamiento de datos para alta carga.
  •	Pruebas Avanzadas:
  ◦	Simular millones de micro-células.
  ◦	Probar fallos de red extensivos.

CoreC Plugins: Extiende el sistema con la flexibilidad de un organismo vivo, integrando nuevas funcionalidades sin límites.

