#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/user_management/utils/db.py
"""
db.py
Maneja la conexión y operaciones con la base de datos user_db para usuarios y roles.
"""

import psycopg2
import json
from typing import Dict, Any
from ...utils.logging import logger

class UserDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.logger = logger.getLogger("UserDB")

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.logger.info("Conectado a user_db")
            return True
        except Exception as e:
            self.logger.error(f"Error conectando a user_db: {e}")
            return False

    async def disconnect(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Desconectado de user_db")

    async def save_user(self, user_id: str, email: str, password: str, name: str, role: str, notification_preferences: Dict, created_at: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO users (user_id, email, password, name, role, notification_preferences, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user_id, email, password, name, role, json.dumps(notification_preferences), created_at)
            )
            self.conn.commit()
            cur.close()
            self.logger.info(f"Usuario {user_id} guardado")
        except Exception as e:
            self.logger.error(f"Error guardando usuario: {e}")
            raise

    async def update_user(self, user_id: str, email: str = None, password: str = None, name: str = None, role: str = None, notification_preferences: Dict = None):
        try:
            cur = self.conn.cursor()
            updates = []
            params = []
            if email:
                updates.append("email = %s")
                params.append(email)
            if password:
                updates.append("password = %s")
                params.append(password)
            if name:
                updates.append("name = %s")
                params.append(name)
            if role:
                updates.append("role = %s")
                params.append(role)
            if notification_preferences:
                updates.append("notification_preferences = %s")
                params.append(json.dumps(notification_preferences))
            if updates:
                params.append(user_id)
                query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s"
                cur.execute(query, params)
                self.conn.commit()
            cur.close()
            self.logger.info(f"Usuario {user_id} actualizado")
        except Exception as e:
            self.logger.error(f"Error actualizando usuario: {e}")
            raise

    async def delete_user(self, user_id: str):
        try:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            self.conn.commit()
            cur.close()
            self.logger.info(f"Usuario {user_id} eliminado")
        except Exception as e:
            self.logger.error(f"Error eliminando usuario: {e}")
            raise

    async def get_user(self, user_id: str) -> Dict[str, Any]:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT user_id, email, name, role, notification_preferences FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            cur.close()
            if user:
                return {
                    "user_id": user[0],
                    "email": user[1],
                    "name": user[2],
                    "role": user[3],
                    "notification_preferences": user[4]
                }
            return None
        except Exception as e:
            self.logger.error(f"Error obteniendo usuario: {e}")
            raise

    async def update_user_preferences(self, user_id: str, notification_preferences: Dict):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE users SET notification_preferences = %s WHERE user_id = %s",
                (json.dumps(notification_preferences), user_id)
            )
            self.conn.commit()
            cur.close()
            self.logger.info(f"Preferencias de notificación actualizadas para {user_id}")
        except Exception as e:
            self.logger.error(f"Error actualizando preferencias: {e}")
            raise