# Plugin ComunicadorInteligente para CoreC

El plugin **ComunicadorInteligente** es una extensión neuronal para CoreC que añade capacidades de **comunicación con el usuario** y **razonamiento avanzado**. Utiliza `openai/gpt-4o-mini` a través de OpenRouter para diálogos naturales y un conjunto de IAs locales (red neuronal, modelo Bayesiano, aprendizaje por refuerzo) para resiliencia, asegurando operación continua incluso si OpenRouter no está disponible. Este plugin actúa como un bloque de LEGO, integrándose perfectamente con CoreC sin modificar su núcleo, manteniendo la ligereza (~100 KB para entidades, ~1 MB para el bloque) y escalabilidad del sistema.

## Características

- **Comunicación Natural**: Procesa consultas de usuarios y genera respuestas inteligentes vía `gpt-4o-mini`.
- **Razonamiento Avanzado**: Analiza datos de CoreC (fitness, anomalías) con IAs locales (red neuronal, Bayesiano, RL).
- **Resiliencia**: Cambia a IAs locales si OpenRouter falla, entrenadas con datos de interacciones previas.
- **Eficiencia**: Usa ~100 entidades ultraligeras (~1 KB cada una) y un bloque simbiótico (~1 MB).
- **Integración Plug-and-Play**: Se registra dinámicamente en CoreC vía `nucleus.py`.
- **Escalabilidad**: Soporta ejecución multi-nodo mediante Redis streams.

## Requisitos

- CoreC (versión compatible, incluido en el repositorio principal)
- Python 3.9+
- Redis (para comunicación via streams)
- Clave API de OpenRouter (obtén en https://openrouter.ai)
- Dependencias: `torch==2.0.1`, `scikit-learn==1.3.2`, `aiohttp==3.9.5`

## Estructura

plugins/comunicador_inteligente/ ├── main.py # Lógica principal (comunicación, razonamiento, resiliencia) ├── config.json # Configuración (canales, OpenRouter, modelos) ├── requirements.txt # Dependencias específicas ├── models/ # Modelos locales │ ├── nn.pt # Red neuronal │ ├── bayes.pkl # Modelo Bayesiano │ └── rl.pkl # Modelo RL (Q-learning) ├── data/ # Datos para entrenamiento │ └── training.log # Registro de interacciones ├── README.md # Este archivo
## Instalación

1. **Asegúrate de tener CoreC configurado**:
   - Sigue las instrucciones en el `README.md` principal de CoreC para configurar el núcleo, Redis, y PostgreSQL.

2. **Instala dependencias del plugin**:
   ```bash
   cd plugins/comunicador_inteligente
   pip install -r requirements.txt
1. Configura OpenRouter:
    * Obtén una clave API en https://openrouter.ai.
    * Actualiza config.json: {
    *     "openrouter_api_key": "tu_api_key_aqui"
    * }
    * 
2. Inicializa modelos locales: mkdir -p models data
3. python -c "from main import RedNeuronalLigera; import torch; nn=RedNeuronalLigera(); torch.save(nn.state_dict(), 'models/nn.pt')"
4. python -c "from sklearn.naive_bayes import GaussianNB; import pickle; bayes=GaussianNB(); pickle.dump(bayes, open('models/bayes.pkl', 'wb'))"
5. python -c "from main import QLearningAgent; import pickle; rl=QLearningAgent(['responder', 'analizar', 'optimizar']); pickle.dump(rl, open('models/rl.pkl', 'wb'))"
6. touch data/training.log
7. 
8. Ejecuta CoreC: cd ../..
9. bash run.sh
10. celery -A corec.core.celery_app worker --loglevel=info
11. 
Uso
1. Enviar consultas:
    * Usa Redis para enviar mensajes al stream user_input_stream: redis-cli XADD user_input_stream * data '{"texto": "Estado del sistema", "valor": 0.5}'
    * 
2. Leer respuestas:
    * Las respuestas se publican en user_output_stream: redis-cli XREAD STREAMS user_output_stream 0-0
    * 
3. Ejemplo de interacción:
    * Entrada: {"texto": "Estado del sistema", "valor": 0.5}
    * Salida esperada: {"texto": "CoreC está operativo, fitness estimado: 0.95", "valor": 0.7}
Pruebas
Ejecuta las pruebas específicas del plugin:
python -m unittest tests/test_comunicador_inteligente.py -v
Notas
* Resiliencia: Si OpenRouter no está disponible, el plugin usa IAs locales entrenadas.
* Escalabilidad: Soporta ejecución en nodos separados via Redis.
* Seguridad: Usa variables de entorno para openrouter_api_key en producción.
Contribuir
Consulta plugins/PLUGINS_GUIDE.md para crear nuevos plugins. Reporta problemas o sugiere mejoras en el repositorio principal de CoreC.
