#!/bin/bash

# run.sh - Script para ejecutar CoreC dentro del proyecto Genesis

# Colores para mensajes en la terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # Sin color

# Función para mostrar mensajes
log() {
    echo -e "${GREEN}[CoreC] $1${NC}"
}

error() {
    echo -e "${RED}[Error] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[Advertencia] $1${NC}"
}

# 1. Verificar que Python 3.10+ esté instalado
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
if [[ -z "$PYTHON_VERSION" ]]; then
    error "Python 3 no está instalado. Por favor, instala Python 3.10 o superior."
    exit 1
fi

if [[ "$(echo $PYTHON_VERSION | grep -oP '^\d+\.\d+')" < "3.10" ]]; then
    error "Se requiere Python 3.10 o superior. Versión actual: $PYTHON_VERSION"
    exit 1
fi

log "Python $PYTHON_VERSION detectado."

# 2. Verificar que las dependencias estén instaladas
if [[ ! -f "requirements.txt" ]]; then
    error "El archivo requirements.txt no existe."
    exit 1
fi

log "Instalando dependencias desde requirements.txt..."
pip3 install -r requirements.txt --quiet
if [[ $? -ne 0 ]]; then
    error "Fallo al instalar las dependencias. Revisa requirements.txt."
    exit 1
fi

# 3. Verificar que Redis esté corriendo
log "Verificando conexión a Redis..."
redis-cli -h localhost -p 6379 ping >/dev/null 2>&1
if [[ $? -ne 0 ]]; then
    error "Redis no está corriendo en localhost:6379. Por favor, inicia Redis."
    exit 1
fi
log "Redis está corriendo."

# 4. Verificar que PostgreSQL esté corriendo
log "Verificando conexión a PostgreSQL..."
PG_PASSWORD=$(grep '"password"' config/corec_config.json | grep -oP '"password":\s*"\K[^"]+')
if ! psql -h localhost -p 5432 -U postgres -d corec_db -c "\q" >/dev/null 2>&1; then
    error "PostgreSQL no está corriendo en localhost:5432 o la base de datos corec_db no existe."
    exit 1
fi
log "PostgreSQL está corriendo."

# 5. Crear tabla 'bloques' si no existe
log "Inicializando la tabla 'bloques' en PostgreSQL..."
python3 -c "
import psycopg2
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CoreCDB')
try:
    conn = psycopg2.connect(dbname='corec_db', user='postgres', password='$PG_PASSWORD', host='localhost', port=5432)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS bloques (
            id VARCHAR(50) PRIMARY KEY,
            canal INTEGER,
            num_entidades INTEGER,
            fitness FLOAT,
            timestamp FLOAT
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()
    logger.info('[DB] Tabla \"bloques\" inicializada')
except Exception as e:
    logger.error(f'[DB] Error inicializando PostgreSQL: {e}')
"

# 6. Ejecutar CoreC
log "Iniciando CoreC..."
python3 run_corec.py

# 7. Mensaje final
log "CoreC detenido."
