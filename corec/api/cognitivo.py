import os
from fastapi import FastAPI, HTTPException, Header
from typing import Dict, Any, List
from corec.nucleus import CoreCNucleus

app = FastAPI(title="CoreC Cognitive API")
API_KEY = os.getenv("API_KEY", "secure_key")  # Cargar desde variable de entorno

nucleus_instance = None

async def get_nucleus():
    global nucleus_instance
    if nucleus_instance is None:
        nucleus_instance = CoreCNucleus("config/corec_config.json")
        await nucleus_instance.inicializar()
    return nucleus_instance

@app.on_event("shutdown")
async def shutdown_event():
    global nucleus_instance
    if nucleus_instance:
        await nucleus_instance.detener()

@app.get("/cognitivo/intuicion/{tipo}")
async def get_intuicion(tipo: str, api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    intuicion = await nucleus.modules["cognitivo"].intuir(tipo)
    return {"tipo": tipo, "intuicion": intuicion}

@app.post("/cognitivo/percibir")
async def percibir(datos: Dict[str, Any], api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    await nucleus.modules["cognitivo"].percibir(datos)
    return {"status": "Percepción registrada", "tipo": datos.get("tipo")}

@app.get("/cognitivo/decisiones")
async def get_decisiones(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    decisiones = nucleus.modules["cognitivo"].decisiones[-10:]
    return {"decisiones": decisiones}

@app.get("/cognitivo/yo")
async def get_yo(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    return {"yo": nucleus.modules["cognitivo"].yo}

@app.get("/cognitivo/metadialogo")
async def get_metadialogo(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    afirmaciones = await nucleus.modules["cognitivo"].generar_metadialogo()
    return {"afirmaciones": afirmaciones}

@app.get("/cognitivo/atencion")
async def get_atencion(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    return {"atencion": nucleus.modules["cognitivo"].atencion}
