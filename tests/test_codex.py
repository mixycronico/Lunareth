# tests/test_codex.py

import sys
import os
import shutil
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure root of project is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plugins.codex.processors.generator import Generator
from plugins.codex.processors.manager import CodexManager
from plugins.codex.processors.schemas import (
    CmdGeneratePlugin,
    CmdGenerateWebsite,
    CmdRevise,
)

@pytest.fixture
def tmp_codex_env(tmp_path, monkeypatch):
    """
    Sets up a temporary templates directory with minimal Jinja2 templates,
    and config dict for Generator and CodexManager.
    """
    # Create templates structure
    tpl_root = tmp_path / "templates"
    plugin_tpl = tpl_root / "plugin"
    react_tpl = tpl_root / "react_app"
    fastapi_tpl = tpl_root / "fastapi_app"
    for d in (plugin_tpl, react_tpl, fastapi_tpl):
        d.mkdir(parents=True)
    # Write a minimal plugin template file
    (plugin_tpl / "file.txt.j2").write_text("Hello {{ plugin_name }}")
    # Write a minimal react template file
    (react_tpl / "index.html.j2").write_text("<h1>{{ project_name }}</h1>")
    # Write a minimal fastapi template file
    (fastapi_tpl / "app.py.j2").write_text("print('{{ project_name }} API')")

    # Configuration for Generator and Manager
    config = {
        "templates_dir": str(tpl_root),
        "output_plugins": str(tmp_path / "out_plugins"),
        "output_websites": str(tmp_path / "out_websites"),
        "exclude_patterns": [],
        "circuit_breaker": {"max_failures": 3, "reset_timeout": 10},
        "stream_in": "in",
        "stream_out": "out",
        "metrics_port": 9001,
        "stream_timeout": 1000,
    }
    # Ensure output directories exist
    (tmp_path / "out_plugins").mkdir()
    (tmp_path / "out_websites").mkdir()
    return config

@pytest.mark.asyncio
async def test_generator_generate_plugin(tmp_codex_env):
    gen = Generator(tmp_codex_env)
    result = await gen.generate_plugin({"plugin_name": "demo"})
    assert result["status"] == "ok"
    # Path should exist and be a directory
    out_path = result["path"]
    assert os.path.isdir(out_path)
    # The rendered file should have .txt extension without .j2 and contain the placeholder
    content = (Path(out_path) / "file.txt").read_text()
    assert content == "Hello demo"

@pytest.mark.asyncio
async def test_generator_generate_react_app(tmp_codex_env):
    gen = Generator(tmp_codex_env)
    result = await gen.generate_website({"template": "react", "project_name": "myapp"})
    assert result["status"] == "ok"
    out_path = result["path"]
    assert os.path.isdir(out_path)
    content = (Path(out_path) / "index.html").read_text()
    assert "<h1>myapp</h1>" in content

@pytest.mark.asyncio
async def test_generator_generate_fastapi_app(tmp_codex_env):
    gen = Generator(tmp_codex_env)
    result = await gen.generate_website({"template": "fastapi", "project_name": "apiapp"})
    assert result["status"] == "ok"
    out_path = result["path"]
    assert os.path.isdir(out_path)
    content = (Path(out_path) / "app.py").read_text()
    assert "print('apiapp API')" in content

@pytest.mark.asyncio
async def test_manager_handle_generate_plugin(tmp_codex_env):
    fake_nucleus = MagicMock()
    fake_nucleus.redis_config = {"username": "", "password": "", "host": "x", "port": 0}
    mgr = CodexManager(fake_nucleus, tmp_codex_env)

    # stub generator
    mgr.generator = MagicMock()
    mgr.generator.generate_plugin = AsyncMock(return_value={"status": "ok", "path": "/fake/plugin"})

    cmd = CmdGeneratePlugin(action="generate_plugin", params={"plugin_name": "test"})
    resp = await mgr.handle(cmd)
    mgr.generator.generate_plugin.assert_awaited_with({"plugin_name": "test"})
    assert resp == {"status": "ok", "path": "/fake/plugin"}

@pytest.mark.asyncio
async def test_manager_handle_generate_website(tmp_codex_env):
    fake_nucleus = MagicMock()
    fake_nucleus.redis_config = {"username": "", "password": "", "host": "x", "port": 0}
    mgr = CodexManager(fake_nucleus, tmp_codex_env)

    # stub generator
    mgr.generator = MagicMock()
    mgr.generator.generate_website = AsyncMock(return_value={"status": "ok", "path": "/fake/website"})

    cmd = CmdGenerateWebsite(action="generate_website", params={"template": "react", "project_name": "site"})
    resp = await mgr.handle(cmd)
    mgr.generator.generate_website.assert_awaited_with({"template": "react", "project_name": "site"})
    assert resp == {"status": "ok", "path": "/fake/website"}

@pytest.mark.asyncio
async def test_manager_handle_revise(tmp_codex_env, tmp_path):
    # prepare a dummy file
    base = Path(tmp_codex_env["templates_dir"]).parent
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    fake_nucleus = MagicMock()
    fake_nucleus.redis_config = {"username": "", "password": "", "host": "x", "port": 0}
    mgr = CodexManager(fake_nucleus, tmp_codex_env)
    # stub reviser
    mgr.reviser = MagicMock()
    mgr.reviser.revisar_codigo = AsyncMock(return_value="print('world')")
    # sanitize_path will resolve within base, so copy to base
    allowed = base / test_file.name
    shutil.copy(str(test_file), str(allowed))
    cmd = CmdRevise(action="revise", params={"file": str(allowed)})
    resp = await mgr.handle(cmd)
    assert resp["status"] == "ok"
    assert "revisado" in resp["message"].lower()
    # file content updated
    assert allowed.read_text() == "print('world')"

class DummyCmd:
    action = "foo"
    params = {}

@pytest.mark.asyncio
async def test_manager_handle_unknown(tmp_codex_env):
    fake_nucleus = MagicMock()
    fake_nucleus.redis_config = {"username": "", "password": "", "host": "x", "port": 0}
    mgr = CodexManager(fake_nucleus, tmp_codex_env)
    resp = await mgr.handle(DummyCmd())
    assert resp["status"] == "error"
    assert "no soportada" in resp["message"].lower()