#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

### Variables: ajusta o exporta antes de ejecutar ###
: "${PG_USER:=postgres}"
: "${PG_PASSWORD:?Debe exportar PG_PASSWORD}"
: "${PG_DB:=corec_db}"
: "${REDIS_PASSWORD:=secure_password}"
: "${CONFIG_PATH:=configs/corec_config.json}"

### Detectar gestor de paquetes ###
if   command -v apt-get  &>/dev/null; then PKG_MGR="apt-get"
elif command -v yum      &>/dev/null; then PKG_MGR="yum"
else  echo "No se detectó apt-get ni yum. Instale manualmente Redis y PostgreSQL."; exit 1; fi

echo "=== 1. Actualizando paquetes y preparando entorno ==="
sudo $PKG_MGR update -y
sudo $PKG_MGR install -y python3 python3-venv python3-pip python3-dev \
                       libpq-dev gcc make \
                       redis-server postgresql postgresql-contrib

echo "=== 2. Configurando Redis ==="
sudo sed -i "s/^# requirepass .*/requirepass ${REDIS_PASSWORD}/" /etc/redis/redis.conf
sudo systemctl enable redis-server
sudo systemctl restart redis-server

echo "=== 3. Configurando PostgreSQL ==="
# Inyecta contraseña y crea base y usuario
sudo -u postgres psql <<-SQL
  ALTER USER ${PG_USER} WITH PASSWORD '${PG_PASSWORD}';
  CREATE DATABASE ${PG_DB};
SQL
sudo systemctl enable postgresql
sudo systemctl restart postgresql

echo "=== 4. Configurando Python virtualenv y dependencias ==="
# Crea entorno virtual en la raíz si no existe
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# Activa e instala
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Instala dependencias de plugins (busca todos los requirements.txt bajo plugins/)
echo "Instalando dependencias de plugins..."
find plugins -maxdepth 2 -name requirements.txt \
  -execdir bash -c 'source "$PWD"/.venv/bin/activate && pip install -r requirements.txt' \;

echo "=== 5. Ajustando corec_config.json ==="
# Reemplaza placeholders con variables de entorno
if grep -q "\${PG_PASSWORD}" "${CONFIG_PATH}"; then
  echo "Reemplazando credenciales en ${CONFIG_PATH}..."
  sed -i "s/\${PG_PASSWORD}/${PG_PASSWORD}/g" "${CONFIG_PATH}"
  sed -i "s/\${REDIS_PASSWORD}/${REDIS_PASSWORD}/g" "${CONFIG_PATH}"
else
  echo "⚠️  Asegúrate de que ${CONFIG_PATH} contiene '\${PG_PASSWORD}' y '\${REDIS_PASSWORD}' placeholders."
fi

echo "=== 6. Inicializando base de datos (particiones) ==="
python3 <<-PY
import psycopg2, time
cfg = ${PG_DB!r}
conn = psycopg2.connect(dbname=${PG_DB!r}, user=${PG_USER!r}, password=${PG_PASSWORD!r})
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS bloques (
  id TEXT PRIMARY KEY,
  canal TEXT,
  num_entidades INTEGER,
  fitness REAL,
  timestamp DOUBLE PRECISION,
  instance_id TEXT
) PARTITION BY RANGE (timestamp);
""")
# Ejemplo: partición para el mes en curso
start = int(time.time())
# next month
end = int(time.time() + 30*24*3600)
cur.execute(f"""
CREATE TABLE IF NOT EXISTS bloques_{time.strftime('%Y_%m')} PARTITION OF bloques
FOR VALUES FROM ({start}) TO ({end});
""")
conn.commit()
cur.close()
conn.close()
PY

echo "=== 7. ¡Listo! ==="
echo "- Redis corriendo en :6379 con contraseña"
echo "- PostgreSQL corriendo en :5432, DB '${PG_DB}'"
echo "- Entorno virtual activo (.venv/)"
echo "- Ejecuta 'bash run.sh' para iniciar CoreC"
echo "- Para tests: 'pytest -q'"