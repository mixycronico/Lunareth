import json
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

def load_secrets(secret_path: str) -> dict:
    with open(secret_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_config() -> dict:
    return load_secrets(Path('configs/core/secrets/db_config.yaml'))

def get_redis_config() -> dict:
    return load_secrets(Path('configs/core/secrets/redis_config.yaml'))

def get_openrouter_config() -> dict:
    return load_secrets(Path('configs/core/secrets/openrouter.yaml'))