#!/bin/bash
# run.sh
echo "Iniciando CoreC..."
# Configura PYTHONPATH para imports
export PYTHONPATH=$PYTHONPATH:./corec:./plugins
# Instala dependencias unificadas
python -m pip install --upgrade pip
pip install -r requirements.txt
# Opcional: ejecutar pruebas localmente
# pytest plugins/crypto_trading/tests/test_crypto_trading.py
# pytest plugins/codex/tests/test_codex.py
# Iniciar CoreC
python -m corec.bootstrap
