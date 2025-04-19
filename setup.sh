#!/bin/bash
# setup.sh
echo "Configurando entorno CoreC..."
pip install -r requirements.txt
echo "Configurando Redis y PostgreSQL..."
echo "Asegúrate de actualizar configs/corec_config.json con las credenciales correctas."