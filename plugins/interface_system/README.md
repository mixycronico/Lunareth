# Plugin InterfaceSystem para CoreC

El plugin **InterfaceSystem** es un panel inteligente que combina una CLI interactiva y una interfaz web, actuando como un asistente conversacional vivo para CoreC, potenciado por **ComunicadorInteligente**. Ofrece comandos para controlar el sistema, monitorear bloques, gestionar plugins, y enviar mensajes al sistema con respuestas de IA. Usa `rich` para una CLI hermosa, `FastAPI` y WebSocket para la web, y una memoria contextual comprimida en Redis.

## Características

- **CLI Hermosa**: Colores, tablas, y banners bioinspirados con `rich`.
- **Interfaz Web**: Chat en tiempo real y controles dinámicos via WebSocket.
- **Comandos Intuitivos**: `status`, `plugins`, `blocks`, `nodes`, `alerts`, `chat`, `activate`, `deactivate`, `config`.
- **Memoria Contextual**: Almacena conversaciones en Redis con compresión `zstd` (fallback a `memory.json`).
- **Sinergia con ComunicadorInteligente**: Respuestas vivas usando `gpt-4o-mini` o IAs locales.
- **Eficiencia**: ~100 entidades (~100 KB), bloque ~1 MB, ~5 MB de memoria.
- **Escalabilidad**: Soporta multi-nodo via Redis.

## Requisitos

- CoreC instalado.
- Python 3.9+.
- Redis 7.0.
- Plugin **ComunicadorInteligente** activo.
- Dependencias: `rich`, `fastapi`, `uvicorn`, `websockets`, `python-socketio`, `click`.

## Estructura

plugins/interface_system/ ├── main.py # CLI principal ├── brain.py # Núcleo conversacional ├── controller.py # Ejecuta acciones en CoreC ├── web_interface.py # Interfaz Web ├── memory.json # Memoria contextual ├── requirements.txt # Dependencias ├── static/ # Archivos estáticos (HTML, CSS, JS) ├── README.md # Este archivo
## Instalación

1. **Instala dependencias**:
   ```bash
   cd plugins/interface_system
   pip install -r requirements.txt
1. Configura CoreC:
    * Verifica CoreC/configs/corec_config.json para Redis y PostgreSQL.
    * Asegúrate de que ComunicadorInteligente esté configurado.
2. Inicializa memoria: touch memory.json
3. 
4. Ejecuta CoreC: cd ../..
5. bash run.sh
6. celery -A corec.core.celery_app worker --loglevel=info
7. 
Uso
CLI
1. Inicia el CLI: python -m plugins.interface_system.main
2. 
3. Comandos disponibles:
    * status: Muestra el estado del sistema.
    * plugins: Lista plugins activos.
    * blocks: Lista bloques simbióticos.
    * nodes: Lista nodos activos.
    * alerts: Muestra alertas recientes.
    * chat : Envía un mensaje al sistema (usa ComunicadorInteligente).
    * activate : Activa un plugin.
    * deactivate : Desactiva un plugin.
    * config : Actualiza una configuración.
4. corec status
5. corec chat "Hola, ¿cuál es el estado?"
6. corec activate comunicador_inteligente
7. 
8. Ejemplo de salida: ╔═══════════════════════════════╗
9. ║ Bienvenido a CoreC Interface  ║
10. ║ Sistema bioinspirado 🌱       ║
11. ╚═══════════════════════════════╝
12. Estado del Sistema
13. ┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
14. ┃ Categoría          ┃ Detalles                 ┃
15. ┣━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━┫
16. ┃ Módulos Activos    ┃ registro, auditoria      ┃
17. ┃ Plugins Activos    ┃ comunicador_inteligente  ┃
18. ┃ Bloques            ┃ 3 activos                ┃
19. ┃ Alertas            ┃ 1 pendientes             ┃
20. ┃ Nodos              ┃ 4                        ┃
21. ┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━┛
22. CoreC: Estado mostrado correctamente 🌱
23. 
Web
1. Inicia el servidor web: uvicorn plugins.interface_system.web_interface:app --host 0.0.0.0 --port 8000
2. 
3. Accede:
    * Abre http://localhost:8000 en un navegador.
    * Usa el chat para enviar comandos (estado, nodos, alertas, activar plugin ).
    * Usa los botones para acciones rápidas.
Pruebas
python -m unittest tests/test_interface_system.py -v
Notas
* Dependencias: Instala todas las dependencias para CLI y web.
* Interacción: Los comandos chat y consultas abiertas requieren ComunicadorInteligente.
* Producción: Usa variables de entorno para credenciales de Redis y OpenRouter.
* Frontend: Personaliza static/ para un diseño más avanzado (por ejemplo, React).
Soporte
Consulta plugins/PLUGIN_DEVELOPMENT.md para crear nuevos plugins.

CoreC: Potenciado por xAI

Integración con CoreC
1. Actualizar requirements.txt: celery==5.3.6
2. redis==5.0.1
3. aioredis==2.0.1
4. psycopg2-binary==2.9.9
5. zstd==1.5.5.1
6. scikit-learn==1.3.2
7. torch==2.0.1
8. jq==1.4.1
9. aiohttp==3.9.5
10. rich==13.5.2
11. fastapi==0.110.0
12. uvicorn==0.29.0
13. websockets==11.0.3
14. python-socketio==5.11.2
15. click==8.1.7
16. 
17. Añadir al README.md de CoreC: ## Plugins
18. 
19. - **ComunicadorInteligente**: Comunicación y razonamiento con `gpt-4o-mini`.
20. - **InterfaceSystem**: CLI y WebSocket para controlar CoreC, potenciado por **ComunicadorInteligente**.
21. 
22. ### Instalación de InterfaceSystem
23. 
24. ```bash
25. cd plugins/interface_system
26. pip install -r requirements.txt
27. touch memory.json
28.  Uso CLI: python -m plugins.interface_system.main
29. corec status
30. corec chat "Hola, ¿cuál es el estado?"
31.  Web: uvicorn plugins.interface_system.web_interface:app --host 0.0.0.0 --port 8000
32.  Accede en http://localhost:8000.
33. 

Uso y Ejemplo
CLI:
python -m plugins.interface_system.main
╔═══════════════════════════════╗
║ Bienvenido a CoreC Interface  ║
║ Sistema bioinspirado 🌱       ║
╚═══════════════════════════════╝
corec status
Estado del Sistema
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Categoría          ┃ Detalles                 ┃
┣━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Módulos Activos    ┃ registro, auditoria      ┃
┃ Plugins Activos    ┃ comunicador_inteligente  ┃
┃ Bloques            ┃ 3 activos                ┃
┃ Alertas            ┃ 1 pendientes             ┃
┃ Nodos              ┃ 4                        ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━┛
CoreC: Estado mostrado correctamente 🌱
corec chat "Hola, ¿cuál es el estado?"
CoreC: CoreC operativo, fitness 0.95 🌟
Web:
* Inicia: uvicorn plugins.interface_system.web_interface:app --host 0.0.0.0 --port 8000
* Abre http://localhost:8000 y usa el chat o botones para interactuar.
