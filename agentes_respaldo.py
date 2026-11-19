"""Ejercicios del notebook (puntos 6 y 7) + traductor con Gemma 3 1B.

IMPORTANTE: el modelo gemma3:1b estándar no tiene llamadas nativas fiables a tools.
Por eso los grafos ejecutan las herramientas de manera explícita: evitan el error
'Ollama ... does not support tools'. El LLM local redacta la respuesta final.
Esta variante es una ORQUESTACIÓN de agentes, no un agente autónomo de tool calling.
"""
import re
import json
import unicodedata
from typing import TypedDict

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from configuracion import RAIZ, crear_modelo
from herramientas import clima, calcular

modelo = crear_modelo()

# ---------------- 6. HERRAMIENTAS Y AGENTES ----------------
class EstadoHerramientas(TypedDict, total=False):
    pregunta: str
    resultados_herramientas: list[str]
    respuesta: str


# Captura expresiones con números y operadores, incluyendo frases frecuentes en español.
NUM = r"-?\d+(?:[.,]\d+)?"
OP = r"(?:\*\*|[+*/×x^]|\-|más|mas|menos|por|entre|dividido\s+por|elevado\s+a)"
PATRON_OPERACIONES = re.compile(rf"({NUM}(?:\s*{OP}\s*{NUM})+)", re.IGNORECASE)
PATRON_ORDENES = re.compile(
    rf"\b(suma|sumar|resta|restar|multiplica|multiplicar|divide|dividir|potencia)\s+({NUM})\s+(?:y|más|mas|menos|por|entre|con|a|elevado\s+a)\s+({NUM})",
    re.IGNORECASE,
)
PATRON_CIUDAD = re.compile(
    r"\b(?:en|de)\s+([a-záéíóúüñ]+(?:\s+[a-záéíóúüñ]+){0,2}?)"
    r"(?=\s*(?:[?¿,;.]|\s+y\b|\s+cu[aá]nto\b|$))",
    flags=re.IGNORECASE,
)


def detectar_ciudad(pregunta: str) -> str | None:
    m = PATRON_CIUDAD.search(pregunta)
    return m.group(1).strip() if m else None


def normalizar_operacion(expresion: str) -> str:
    # Solo se normalizan cadenas numéricas reconocidas por expresiones regulares.
    expresion = expresion.replace(",", ".")
    expresion = re.sub(r"\bdividido\s+por\b|\bentre\b", "/", expresion, flags=re.I)
    expresion = re.sub(r"\belevado\s+a\b|\^", "**", expresion, flags=re.I)
    expresion = re.sub(r"\bm[aá]s\b", "+", expresion, flags=re.I)
    expresion = re.sub(r"\bmenos\b", "-", expresion, flags=re.I)
    expresion = re.sub(r"\bpor\b|[×x]", "*", expresion, flags=re.I)
    return expresion


def obtener_operaciones(pregunta: str) -> list[str]:
    operaciones = []
    for m in PATRON_ORDENES.finditer(pregunta):
        verbo, a, b = m.group(1).lower(), m.group(2), m.group(3)
        operador = {"suma": "+", "sumar": "+", "resta": "-", "restar": "-",
                    "multiplica": "*", "multiplicar": "*", "divide": "/",
                    "dividir": "/", "potencia": "**"}[verbo]
        operaciones.append((m.start(), f"{a}{operador}{b}"))
    for m in PATRON_OPERACIONES.finditer(pregunta):
        if any(a <= m.start() < b for a, b in ((k.start(), k.end()) for k in PATRON_ORDENES.finditer(pregunta))):
            continue
        operaciones.append((m.start(), normalizar_operacion(m.group(1))))
    # Evita duplicados cuando una frase coincide con varios patrones.
    return list(dict.fromkeys(e for _, e in sorted(operaciones)))


def ejecutar_herramientas(estado: EstadoHerramientas):
    pregunta = estado["pregunta"]
    resultados: list[str] = []
    if re.search(r"\b(clima|tiempo|temperatura|pron[oó]stico)\b", pregunta, re.I):
        ciudad = detectar_ciudad(pregunta)
        resultados.append(clima.invoke({"ciudad": ciudad}) if ciudad
                          else "Falta la ciudad para consultar el clima; pídela al usuario.")

    for expresion in obtener_operaciones(pregunta):
        valor = calcular.invoke({"expresion": expresion})
        resultados.append(f"Cálculo {expresion} = {valor}.")

    if not resultados:
        resultados.append("NO_HERRAMIENTAS: Pregunta por el clima en una ciudad o una operación aritmética (ej. 8+5, 10/2, 2^3).")
    return {"resultados_herramientas": resultados}


prompt_herramientas = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente en español. Resume brevemente los RESULTADOS DE "
               "HERRAMIENTAS suministrados. Los números y mediciones deben copiarse "
               "tal como están. No inventes valores ni supongas que datos simulados son reales."),
    ("human", "Solicitud: {pregunta}\nResultados de herramientas:\n{resultados}"),
])
cadena_herramientas = prompt_herramientas | modelo | StrOutputParser()


def responder_herramientas(estado: EstadoHerramientas):
    resultados = estado["resultados_herramientas"]
    if resultados == ["NO_HERRAMIENTAS: Pregunta por el clima en una ciudad o una operación aritmética (ej. 8+5, 10/2, 2^3)."]:
        return {"respuesta": "Pídeme clima en una ciudad o un cálculo, por ejemplo 7+3, 20/4 o 2^5."}
    return {"respuesta": cadena_herramientas.invoke({
        "pregunta": estado["pregunta"], "resultados": "\n".join(resultados)
    })}


def crear_agente_herramientas():
    grafo = StateGraph(EstadoHerramientas)
    grafo.add_node("usar_herramientas", ejecutar_herramientas)
    grafo.add_node("responder_con_gemma", responder_herramientas)
    grafo.add_edge(START, "usar_herramientas")
    grafo.add_edge("usar_herramientas", "responder_con_gemma")
    grafo.add_edge("responder_con_gemma", END)
    return grafo.compile()


agente_herramientas = crear_agente_herramientas()


@traceable(name="Punto_6_agente_clima_calculadora", run_type="chain")
def consultar_herramientas(pregunta: str) -> dict:
    return agente_herramientas.invoke({"pregunta": pregunta})


# ---------------- 7. RAG FAQ MERCADO LIBRE COLOMBIA ----------------
# Material local para que el ejercicio funcione offline: no consulta la web en vivo.
RUTA_FAQ = RAIZ / "datos" / "faq_mercadolibre_colombia.json"
faq = json.loads(RUTA_FAQ.read_text(encoding="utf-8"))
documentos = [Document(page_content=f"Pregunta: {f['pregunta']}\nRespuesta: {f['respuesta']}",
                       metadata={"fuente": f["fuente"], "pregunta": f["pregunta"]}) for f in faq]

STOPWORDS = {
    "a", "al", "con", "como", "cual", "cuales", "cuanto", "cuantos", "de", "del",
    "el", "en", "es", "esta", "hace", "hacer", "hay", "la", "las", "lo", "los",
    "me", "mi", "mis", "para", "por", "puedo", "que", "se", "si", "son", "su",
    "tengo", "un", "una", "unos", "unas", "y", "yo", "mercado", "libre", "compra",
    "compras", "producto", "productos", "quiero", "puede", "pueden", "colombia",
}


def palabras_relevantes(texto: str) -> set[str]:
    normalizado = unicodedata.normalize("NFD", texto.lower())
    normalizado = "".join(c for c in normalizado if unicodedata.category(c) != "Mn")
    return {p for p in re.findall(r"[a-z0-9]+", normalizado)
            if p not in STOPWORDS and len(p) > 2}


def tokenizar(texto: str) -> list[str]:
    return sorted(palabras_relevantes(texto))


retriever = BM25Retriever.from_documents(documentos, preprocess_func=tokenizar, k=3)


@tool
def buscar_faq_mercadolibre(consulta: str) -> str:
    """Busca respuestas frecuentes de Mercado Libre Colombia en una base local con enlaces oficiales."""
    palabras = palabras_relevantes(consulta)
    if not palabras:
        return "SIN_EVIDENCIA"
    candidatos = retriever.invoke(consulta)
    pertinentes = [d for d in candidatos if palabras & palabras_relevantes(d.page_content)]
    if not pertinentes:
        return "SIN_EVIDENCIA"
    return "\n\n".join(f"[Fuente: {d.metadata['fuente']}] {d.page_content}"
                       for d in pertinentes[:2])


class EstadoRAG(TypedDict, total=False):
    pregunta: str
    contexto: str
    respuesta: str


def recuperar(estado: EstadoRAG):
    return {"contexto": buscar_faq_mercadolibre.invoke({"consulta": estado["pregunta"]})}


prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente académico de preguntas frecuentes de Mercado Libre Colombia. "
               "Responde SOLO con lo que contiene el CONTEXTO de páginas oficiales. "
               "Estas son copias locales resumidas, NO datos en tiempo real. "
               "Si la información no alcanza, responde: 'No tengo esa información en las FAQ consultadas'. "
               "Conserva condiciones y excepciones; no inventes políticas ni detalles de compras personales."),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
])
cadena_rag = prompt_rag | modelo | StrOutputParser()


def responder_rag(estado: EstadoRAG):
    if estado["contexto"] == "SIN_EVIDENCIA":
        return {"respuesta": "No tengo esa información en las FAQ consultadas."}
    respuesta = cadena_rag.invoke({"contexto": estado["contexto"], "pregunta": estado["pregunta"]})
    urls = list(dict.fromkeys(re.findall(r"\[Fuente: (https?://[^\]]+)\]", estado["contexto"])))
    return {"respuesta": respuesta.strip() + "\n\nFuentes: " + ", ".join(urls) +
            "\n(Base FAQ local; verifica cambios actuales en las páginas oficiales.)"}


def crear_agente_rag():
    grafo = StateGraph(EstadoRAG)
    grafo.add_node("buscar_faq_BM25", recuperar)
    grafo.add_node("responder_con_gemma", responder_rag)
    grafo.add_edge(START, "buscar_faq_BM25")
    grafo.add_edge("buscar_faq_BM25", "responder_con_gemma")
    grafo.add_edge("responder_con_gemma", END)
    return grafo.compile()


agente_rag = crear_agente_rag()


@traceable(name="Punto_7_RAG_FAQ_MercadoLibre", run_type="chain")
def consultar_rag(pregunta: str) -> dict:
    return agente_rag.invoke({"pregunta": pregunta})


# ---------------- 8. TRADUCTOR (ADICIÓN DEL ESTUDIANTE) ----------------
prompt_traductor = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor profesional. Traduce del idioma {origen} al {destino}. "
               "Devuelve SOLAMENTE el texto traducido, sin comentarios. Conserva nombres, "
               "cifras, tono y significado."),
    ("human", "{texto}"),
])
cadena_traductor = prompt_traductor | modelo | StrOutputParser()


@traceable(name="Agente_traductor_con_Gemma", run_type="chain")
def traducir(texto: str, origen: str = "español", destino: str = "inglés") -> str:
    return cadena_traductor.invoke({"texto": texto, "origen": origen, "destino": destino})
