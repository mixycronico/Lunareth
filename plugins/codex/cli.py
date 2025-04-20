#!/usr/bin/env python3
# plugins/codex/cli.py
import json, click, asyncio
from aioredis import from_url

STREAM = "corec_commands"

async def send(cmd):
    r = await from_url("redis://localhost:6379", decode_responses=True)
    await r.xadd(STREAM, {"data": json.dumps(cmd)})
    # esperar respuesta
    data = await r.xread({"corec_responses":"0-0"}, count=1, block=5000)
    print("Respuesta:", data)

@click.group()
def cli():
    """CLI Codex — genera plugins y webs al vuelo."""
    pass

@cli.command()
@click.argument("plugin_name", metavar="<plugin_name>")
def generate_plugin(plugin_name):
    """Genera un nuevo plugin CoreC."""
    cmd = {"action":"generate_plugin","params":{"plugin_name":plugin_name}}
    asyncio.run(send(cmd))

@cli.command()
@click.argument("template", type=click.Choice(["react","fastapi"]))
@click.argument("project_name", metavar="<project_name>")
def generate_website(template, project_name):
    """Genera una aplicación web."""
    cmd = {"action":"generate_website","params":{"template":template,"project_name":project_name}}
    asyncio.run(send(cmd))

@cli.command()
@click.argument("file_path", metavar="<file_path>")
def revise(file_path):
    """Refactoriza un archivo existente."""
    cmd = {"action":"revise","params":{"file":file_path}}
    asyncio.run(send(cmd))

# Autocompletado
try:
    import click_completion
    click_completion.init()
except ImportError:
    pass

if __name__ == "__main__":
    cli()