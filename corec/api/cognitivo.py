from fastapi import FastAPI, HTTPException, Header
from typing import Dict, Any
from corec.nucleus import CoreCNucleus

app = FastAPI(title="CoreC Cognitive API")

# Única instancia compartida del núcleo
_nucleus_instance: CoreCNucleus = None


async def get_nucleus() -> CoreCNucleus:
    """Inicializa y devuelve la instancia singleton de CoreCNucleus."""
    global _nucleus_instance
    if _nucleus_instance is None:
        _nucleus_instance = CoreCNucleus("config/corec_config.json")
        await _nucleus_instance.inicializar()
    return _nucleus_instance


@app.on_event("shutdown")
async def shutdown_event():
    """Detiene el núcleo al apagar la aplicación."""
    global _nucleus_instance
    if _nucleus_instance is not None:
        await _nucleus_instance.detener()


@app.get("/cognitivo/intuicion/{tipo}")
async def get_intuicion(tipo: str, api_key: str = Header(...)):
    if api_key != "secure_key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    intuicion = await nucleus.modules["cognitivo"].intuir(tipo)
    return {"tipo": tipo, "intuicion": intuicion}


@app.post("/cognitivo/percibir")
async def percibir(datos: Dict[str, Any], api_key: str = Header(...)):
    if api_key != "secure_key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    await nucleus.modules["cognitivo"].percibir(datos)
    return {"status": "Percepción registrada", "tipo": datos.get("tipo")}


@app.get("/cognitivo/decisiones")
async def get_decisiones(api_key: str = Header(...)):
    if api_key != "secure_key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    decisiones = nucleus.modules["cognitivo"].decisiones[-10:]
    return {"decisiones": decisiones}


@app.get("/cognitivo/yo")
async def get_yo(api_key: str = Header(...)):
    if api_key != "secure_key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    return {"yo": nucleus.modules["cognitivo"].yo}


@app.get("/cognitivo/metadialogo")
async def get_metadialogo(api_key: str = Header(...)):
    if api_key != "secure_key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    afirmaciones = await nucleus.modules["cognitivo"].generar_metadialogo()
    return {"afirmaciones": afirmaciones}


@app.get("/cognitivo/atencion")
async def get_atencion(api_key: str = Header(...)):
    if api_key != "secure_key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    nucleus = await get_nucleus()
    return {"atencion": nucleus.modules["cognitivo"].atencion}
