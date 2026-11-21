"""API local para exponer los agentes de Agentes_Gemma_MercadoLibre a Lovable.

Copiar a la RAIZ del proyecto (junto a agentes.py y configuracion.py).
Instalar dependencias con: python -m pip install -r requirements_api.txt
Levantar con: python -m uvicorn api:app --host 127.0.0.1 --port 8000
"""
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configuracion import MODELO_NOMBRE
from agentes import consultar_herramientas, consultar_rag, traducir

app = FastAPI(title="Gemma - API de agentes", version="1.0.0")

# Exponer solo a los orígenes locales de desarrollo; NO usar '*' para una API pública.
allowed_origins = [
    o.strip() for o in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Solicitud(BaseModel):
    agente: Literal["herramientas", "rag", "traductor"]
    pregunta: str = Field(min_length=1, max_length=4000)
    origen: str = Field(default="español", max_length=80)
    destino: str = Field(default="inglés", max_length=80)


class Respuesta(BaseModel):
    agente: str
    respuesta: str
    detalles: list[str] = Field(default_factory=list)
    fuentes: list[str] = Field(default_factory=list)
    modelo: str


def _fuentes_desde_contexto(contexto: str) -> list[str]:
    import re
    return list(dict.fromkeys(re.findall(r"\[Fuente:\s*(https?://[^\]]+)\]", contexto)))


@app.get("/salud")
def salud():
    """Comprueba que FastAPI está activo; no garantiza disponibilidad de Ollama."""
    return {"estado": "ok", "modelo": MODELO_NOMBRE}


@app.get("/agentes")
def agentes_disponibles():
    return {"agentes": ["herramientas", "rag", "traductor"], "modelo": MODELO_NOMBRE}


@app.post("/preguntar", response_model=Respuesta)
def preguntar(s: Solicitud):
    try:
        if s.agente == "herramientas":
            result = consultar_herramientas(s.pregunta)
            return Respuesta(
                agente=s.agente,
                respuesta=str(result["respuesta"]),
                detalles=[str(v) for v in result.get("resultados_herramientas", [])],
                modelo=MODELO_NOMBRE,
            )
        if s.agente == "rag":
            result = consultar_rag(s.pregunta)
            return Respuesta(
                agente=s.agente,
                respuesta=str(result["respuesta"]),
                fuentes=_fuentes_desde_contexto(str(result.get("contexto", ""))),
                modelo=MODELO_NOMBRE,
            )
        return Respuesta(
            agente=s.agente,
            respuesta=str(traducir(texto=s.pregunta, origen=s.origen, destino=s.destino)),
            modelo=MODELO_NOMBRE,
        )
    except Exception as exc:
        # Evitar enviar trazas técnicas o variables privadas al navegador.
        raise HTTPException(
            status_code=503,
            detail="El agente no pudo responder. Verifica que Ollama esté activo y que el modelo esté instalado.",
        ) from exc
