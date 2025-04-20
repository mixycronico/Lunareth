# tests/test_corec.py
import pytest
import asyncio
import json
import struct
import random
import time
from pathlib import Path

import psycopg2
import redis.asyncio as aioredis
import zstd

from corec.serialization import serializar_mensaje, deserializar_mensaje, MESSAGE_FORMAT
from corec.db import init_postgresql
from corec.redis_client import init_redis
from corec.entities import (
    crear_entidad, procesar_entidad,
    crear_celu_entidad, procesar_celu_entidad
)
from corec.blocks import BloqueSimbiotico
from corec.processors import ProcesadorSensor, ProcesadorFiltro
from corec.core import cargar_config, ModuloBase
from corec.nucleus import CoreCNucleus
from corec.bootstrap import main as bootstrap_main
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion

@pytest.mark.asyncio
async def test_serialization_roundtrip():
    id_, canal, valor, activo = 42, 3, 1.2345, True
    raw = await serializar_mensaje(id_, canal, valor, activo)
    assert isinstance(raw, (bytes, bytearray))
    out = await deserializar_mensaje(raw)
    assert out["id"] == id_
    assert out["canal"] == canal
    assert pytest.approx(out["valor"], rel=1e-6) == valor
    assert out["activo"] is True

def test_message_format_size():
    # struct "!Ibf?" debe ocupar 4+1+4+1 = 10 bytes
    assert struct.calcsize(MESSAGE_FORMAT) == 10

def test_cargar_config(tmp_path):
    # Test carga de JSON válido
    cfg = {"instance_id":"t1","db_config":{}, "redis_config":{}}
    p = tmp_path/"cfg.json"
    p.write_text(json.dumps(cfg))
    loaded = cargar_config(str(p))
    assert loaded == cfg

def test_init_postgresql(monkeypatch):
    # Monkeypatch psycopg2.connect para capturar queries
    calls = []
    class DummyCursor:
        def execute(self, q, *args, **kwargs):
            calls.append(q)
        def close(self): pass
    class DummyConn:
        def __init__(self, **kw): self.cur = DummyCursor()
        def cursor(self): return self.cur
        def commit(self): pass
        def close(self): pass

    monkeypatch.setattr(psycopg2, "connect", lambda **kw: DummyConn())
    init_postgresql({
        "dbname":"db", "user":"u", "password":"p", "host":"h", "port":1
    })
    assert any("CREATE TABLE IF NOT EXISTS bloques" in q for q in calls)

@pytest.mark.asyncio
async def test_init_redis(monkeypatch):
    # Monkeypatch aioredis.from_url
    dummy = object()
    async def fake_from_url(url, decode_responses):
        return dummy

    monkeypatch.setattr(aioredis, "from_url", fake_from_url)
    client = await init_redis({
        "host":"h", "port":1, "username":"u", "password":"p"
    })
    assert client is dummy

@pytest.mark.asyncio
async def test_micro_entity_processing():
    async def f(): return {"valor": 0.8}
    ent = crear_entidad("m10", 2, f)
    raw = await procesar_entidad(ent, umbral=0.5)
    out = await deserializar_mensaje(raw)
    assert out["id"] == 10
    assert out["canal"] == 2
    assert out["activo"] is True
    assert pytest.approx(out["valor"], rel=1e-6) == 0.8

@pytest.mark.asyncio
async def test_micro_entity_inactive():
    async def f(): return {"valor": 0.9}
    ent = crear_entidad("m11", 3, f)
    ent_inactive = (ent[0], ent[1], ent[2], False)
    raw = await procesar_entidad(ent_inactive)
    out = await deserializar_mensaje(raw)
    assert out["activo"] is False
    assert out["valor"] == 0.0

@pytest.mark.asyncio
async def test_celu_entity_processing_and_error():
    async def proc_ok(d): return {"valor": d.get("x",0)+0.2}
    ent_ok = crear_celu_entidad("c5", 1, proc_ok)
    raw_ok = await procesar_celu_entidad(ent_ok, {"x":0.3}, umbral=0.4)
    out_ok = await deserializar_mensaje(raw_ok)
    assert out_ok["activo"] is True
    assert pytest.approx(out_ok["valor"], rel=1e-6) == 0.5

    async def proc_err(d): raise RuntimeError("boom")
    ent_err = crear_celu_entidad("c6", 1, proc_err)
    raw_err = await procesar_celu_entidad(ent_err, {}, umbral=0.1)
    out_err = await deserializar_mensaje(raw_err)
    assert out_err["activo"] is False
    assert out_err["valor"] == 0.0

@pytest.mark.asyncio
async def test_processors():
    ps = ProcesadorSensor()
    pf = ProcesadorFiltro()
    r1 = await ps.procesar({"valores":[1.0,2.0,3.0]})
    assert r1["valor"] == pytest.approx(2.0)
    r2 = await pf.procesar({"valor":0.6, "umbral":0.5})
    assert r2["valor"] == pytest.approx(0.6)
    r3 = await pf.procesar({"valor":0.4, "umbral":0.5})
    assert r3["valor"] == pytest.approx(0.0)

@pytest.mark.asyncio
async def test_block_processing_and_write(monkeypatch):
    # stub DB + compresión
    class DummyCursor:
        def __init__(self): self.queries=[]
        def execute(self, q, params=None): self.queries.append(q)
        def close(self): pass
    class DummyConn:
        def __init__(self, **kw): self.cur = DummyCursor()
        def cursor(self): return self.cur
        def commit(self): pass
        def close(self): pass

    monkeypatch.setattr(psycopg2, "connect", lambda **kw: DummyConn())
    monkeypatch.setattr(zstd, "compress", lambda data, level: data)

    async def f1(): return {"valor":1.0}
    async def f2(): return {"valor":0.0}
    ents = [crear_entidad("m1",1,f1), crear_entidad("m2",1,f2)]
    stub = type("N",(object,),{"instance_id":"X"})()
    bloq = BloqueSimbiotico("B1",1,ents,max_size_mb=1,nucleus=stub)

    out = await bloq.procesar(1.0)
    assert out["bloque_id"] == "B1"
    assert isinstance(out["mensajes"], list)

    await bloq.escribir_postgresql({"dbname":"x","user":"u","password":"p","host":"h","port":1})
    assert any("INSERT INTO bloques" in q for q in bloq.nucleus is not None and bloq.nucleus)

@pytest.mark.asyncio
async def test_nucleus_and_modules_loading(monkeypatch, tmp_path):
    # config temporal
    cfg = {"instance_id":"t","db_config":{},"redis_config":{},"bloques":[]}
    cf = tmp_path/"cfg.json"
    cf.write_text(json.dumps(cfg))

    # stub DB y Redis
    monkeypatch.setattr("corec.db.init_postgresql", lambda c: None)
    async def fake_redis(c): return "R"
    monkeypatch.setattr("corec.redis_client.init_redis", fake_redis)

    nuc = CoreCNucleus(str(cf))
    await nuc.inicializar()
    assert nuc.redis_client == "R"
    # debe cargar los 4 módulos básicos
    assert set(nuc.modules.keys()) == {"registro","ejecucion","auditoria","sincronizacion"}

@pytest.mark.asyncio
async def test_modulo_registro_and_sincronizacion():
    # Registro
    mod = ModuloRegistro()
    nucleus = type("N",(object,),{"config":{"bloques":[{"id":"b1","canal":1,"entidades":10}]}})()
    await mod.inicializar(nucleus)
    assert "b1" in mod.bloques

    # Sincronización
    mod2 = ModuloSincronizacion()
    nucleus2 = type("N",(object,),{"modules":{"registro": mod}})()
    # crear dos bloques simulados
    b1 = BloqueSimbiotico("b1",1,[],max_size_mb=1,nucleus=nucleus2)
    b2 = BloqueSimbiotico("b2",1,[],max_size_mb=1,nucleus=nucleus2)
    mod.bloques = {"b1":b1, "b2":b2}
    nucleus2.modules = {"registro": mod}
    await mod2.inicializar(nucleus2)
    await mod2.fusionar_bloques("b1","b2","b3")
    assert "b3" in mod.bloques

# (Opcional) Prueba mínima de arranque
@pytest.mark.asyncio
async def test_bootstrap_runs(monkeypatch, tmp_path):
    # evita cargar módulos reales
    monkeypatch.setattr("corec.bootstrap.CoreCNucleus", lambda cfg: type("X",(object,),{
        "inicializar": asyncio.coroutine(lambda self: None),
        "ejecutar": asyncio.coroutine(lambda self: None),
        "detener": asyncio.coroutine(lambda self: None)
    })())
    # solo llama a main y no falle
    await bootstrap_main()