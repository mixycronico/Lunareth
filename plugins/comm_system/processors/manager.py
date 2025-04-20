#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/comm_system/processors/manager.py
Loop principal que despacha comandos de Redis Streams.
"""
import asyncio
import json
import logging
from corec.core import aioredis
from typing import Dict, Any
from .memory import ChatMemory
from .ai_chat import AIChat

class CommManager:
    def __init__(self, nucleus, config):
        self.nucleus    = nucleus
        self.config     = config
        self.logger     = logging.getLogger("CommManager")
        self.redis      = None
        self.chat_ai    = AIChat(config)
        self.memory     = None
        self.stream_in  = config["stream_in"]
        self.stream_out = config["stream_out"]

    async def init_redis(self):
        url = (
            f"redis://{self.nucleus.redis_config['username']}:"
            f"{self.nucleus.redis_config['password']}@"
            f"{self.nucleus.redis_config['host']}:"
            f"{self.nucleus.redis_config['port']}"
        )
        self.redis  = await aioredis.from_url(url, decode_responses=False)
        # inicializa memoria
        self.memory = ChatMemory(self.redis,
                                 ttl=self.config["memory_ttl"],
                                 disk_path=self.config["memory_file"])
        await self.memory.load()

    async def run_loop(self):
        last_id = "0-0"
        while True:
            entries = await self.redis.xread({self.stream_in: last_id},
                                             count=1, block=5000)
            if not entries:
                continue
            _, msgs = entries[0]
            for msg_id, fields in msgs:
                cmd = json.loads(fields["data"])
                resp = await self.handle_command(cmd)
                await self.redis.xadd(self.stream_out,
                                      {"data": json.dumps(resp)})
                last_id = msg_id

    async def handle_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        action = cmd.get("action")
        params = cmd.get("params", {})

        if action == "chat":
            question     = params.get("mensaje", "")
            history      = await self.memory.load()
            context_msgs = self.memory.to_openai(history)
            answer       = await self.chat_ai.chat(context_msgs, question)
            await self.memory.add_pair(question, answer)
            return {"status":"ok", "texto": answer}

        if action == "create_plugin":
            name = params.get("plugin_name")
            # aquí iría tu generator (no incluido aún)
            return {"status":"error", "message":"Función create_plugin no implementada"}

        if action == "status":
            mods  = list(self.nucleus.modulos.keys())
            plugs = list(self.nucleus.plugins.keys())
            return {"status":"ok", "modulos": mods, "plugins": plugs}

        return {"status":"error", "message": f"Acción desconocida '{action}'."}

    async def teardown(self):
        # Limpieza si es necesaria
        pass