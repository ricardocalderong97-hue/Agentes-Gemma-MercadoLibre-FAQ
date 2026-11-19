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


# Reconocimiento de expresiones aritméticas con lenguaje natural y paréntesis.
# El cálculo se delega a la herramienta segura calcular (AST de Python).
NUM = r"-?\d+(?:[.,]\d+)?"
PATRON_ORDENES = re.compile(
    rf"\b(suma|sumar|resta|restar|multiplica|multiplicar|divide|dividir|potencia)\s+({NUM})\s+(?:y|más|mas|menos|por|entre|con|a|elevado\s+a)\s+({NUM})",
    re.IGNORECASE,
)
PATRON_CIUDAD = re.compile(
    r"\b(?:en|de)\s+([a-záéíóúüñ]+(?:\s+[a-záéíóúüñ]+){0,2}?)"
    r"(?=\s*(?:[?¿,;.]|\s+y\b|\s+cu[aá]nto\b|$))",
    flags=re.IGNORECASE,
)
# Acepta expresiones completas, por ejemplo (8+4)*3, 10/2, 2**5.
PATRON_EXPRESION = re.compile(
    rf"(?<![\w])(?:\(\s*)?{NUM}"
    rf"(?:\s*\)*\s*(?:\*\*|[+\-*/^])\s*\(*\s*{NUM})+\s*\)*"
)


def detectar_ciudad(pregunta: str) -> str | None:
    m = PATRON_CIUDAD.search(pregunta)
    return m.group(1).strip() if m else None


def normalizar_operacion(texto: str) -> str:
    texto = re.sub(r"\bdividido\s+(?:entre|por)\b", "/", texto, flags=re.I)
    texto = re.sub(r"\belevado\s+a\b", "**", texto, flags=re.I)
    texto = re.sub(r"\b(?:más|mas)\b", "+", texto, flags=re.I)
    texto = re.sub(r"\bmenos\b", "-", texto, flags=re.I)
    # 'entre' significa división; 'por' significa multiplicación.
    texto = re.sub(r"\bentre\b", "/", texto, flags=re.I)
    texto = re.sub(r"\bpor\b|×", "*", texto, flags=re.I)
    texto = texto.replace("^", "**")
    return texto


def obtener_operaciones(pregunta: str) -> list[str]:
    # Cambia palabras de operación en toda la frase, preservando el resto.
    texto = normalizar_operacion(pregunta)
    operaciones: list[tuple[int, str]] = []
    ordenes = list(PATRON_ORDENES.finditer(pregunta))
    for m in ordenes:
        verbo, a, b = m.group(1).lower(), m.group(2), m.group(3)
        operador = {"suma": "+", "sumar": "+", "resta": "-", "restar": "-",
                    "multiplica": "*", "multiplicar": "*", "divide": "/",
                    "dividir": "/", "potencia": "**"}[verbo]
        operaciones.append((m.start(), f"{a}{operador}{b}"))

    # Ubica las expresiones completas directamente en el texto normalizado.
    # Los reemplazos pueden cambiar las posiciones; se evita repetir por valor.
    for m in PATRON_EXPRESION.finditer(texto):
        expresion = m.group().strip()
        # No confundir puntuación final ni una coma separadora con decimales.
        expresion = expresion.rstrip(" .,")
        # Si sobra un paréntesis de cierre, calcular() lo señalará como error.
        if expresion:
            operaciones.append((m.start(), expresion.replace(",", ".")))
    return list(dict.fromkeys(op for _, op in operaciones))


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
    respuesta_modelo = cadena_herramientas.invoke({
        "pregunta": estado["pregunta"], "resultados": "\n".join(resultados)
    })
    if len(resultados) > 1:
        # Garantiza que el texto final siempre incluya TODOS los resultados,
        # incluso si Gemma omite alguno en su resumen.
        return {"respuesta": "\n".join(resultados) + "\n\nResumen de Gemma: " + respuesta_modelo}
    return {"respuesta": respuesta_modelo}


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


# Se recupera todo el conjunto pequeno de FAQs (12 entradas) antes de reordenar.
# BM25 aporta un ranking inicial, y la coincidencia con la PREGUNTA de cada FAQ
# evita confundir preguntas diferentes sobre un mismo tema (por ejemplo, pagos).
retriever = BM25Retriever.from_documents(documentos, preprocess_func=tokenizar, k=len(documentos))


def normalizar_pregunta(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", texto))


def puntuar_faq(consulta: str, doc: Document) -> float:
    """Prioriza la pregunta equivalente y no solo el tema general del documento."""
    pregunta_faq = doc.metadata["pregunta"]
    tokens_usuario = palabras_relevantes(consulta)
    tokens_faq = palabras_relevantes(pregunta_faq)
    if not tokens_usuario or not tokens_faq:
        return 0.0
    compartidos = tokens_usuario & tokens_faq
    if not compartidos:
        return 0.0
    precision = len(compartidos) / len(tokens_faq)
    cobertura = len(compartidos) / len(tokens_usuario)
    # F1 distingue similitud bilateral (y evita que un solo termino, 'pago',
    # desplace la pregunta con tres terminos pertinentes).
    score = 2 * precision * cobertura / (precision + cobertura)
    q = normalizar_pregunta(consulta)
    p = normalizar_pregunta(pregunta_faq)
    if q == p:
        score += 2.0
    elif len(compartidos) >= 2 and (q in p or p in q):
        score += 0.3
    return score


@tool
def buscar_faq_mercadolibre(consulta: str) -> str:
    """Recupera las FAQ locales con BM25 y reordenamiento por pregunta específica."""
    palabras = palabras_relevantes(consulta)
    if not palabras:
        return "SIN_EVIDENCIA"
    # BM25 permite obtener candidatos; reordenar usa el titulo/pregunta
    # de las FAQ y evita mezclar distintos metodos de pago.
    candidatos = retriever.invoke(consulta)
    ordenados = sorted(candidatos, key=lambda d: puntuar_faq(consulta, d), reverse=True)
    if not ordenados or puntuar_faq(consulta, ordenados[0]) < 0.48:
        return "SIN_EVIDENCIA"
    # Incluir SOLO la mejor FAQ: evita que Gemma mezcle politicas de otras compras.
    principal = ordenados[0]
    return f"[Fuente: {principal.metadata['fuente']}] {principal.page_content}"


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
               "Conserva condiciones y excepciones, incluyendo el método o canal de pago "
               "(por ejemplo PSE) cuando aparezca en el contexto. "
               "No inventes políticas ni detalles de compras personales."),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
])
cadena_rag = prompt_rag | modelo | StrOutputParser()


def preservar_datos_clave(respuesta: str, contexto: str) -> str:
    """Evita omisiones comprobables de siglas y cantidades en la FAQ recuperada.

    No genera información nueva: si Gemma omite una sigla como PSE o un
    plazo como 7 días, se añade el texto canónico de la FAQ local.
    """
    if "Respuesta: " not in contexto:
        return respuesta.strip()
    respuesta_fuente = contexto.split("Respuesta: ", 1)[1].strip()
    # El contenido recuperado de la FAQ es de una sola línea.
    respuesta_fuente = respuesta_fuente.splitlines()[0].strip()
    siglas = set(re.findall(r"\b[A-Z]{2,}\b", respuesta_fuente))
    cantidades = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", respuesta_fuente))
    faltan_siglas = any(sigla not in respuesta.upper() for sigla in siglas)
    faltan_cantidades = any(not re.search(rf"(?<!\d){re.escape(n)}(?!\d)", respuesta)
                           for n in cantidades)
    if faltan_siglas or faltan_cantidades:
        return respuesta.strip() + "\n\nDetalle respaldado por la FAQ: " + respuesta_fuente
    return respuesta.strip()


def responder_rag(estado: EstadoRAG):
    if estado["contexto"] == "SIN_EVIDENCIA":
        return {"respuesta": "No tengo esa información en las FAQ consultadas."}
    respuesta = cadena_rag.invoke({"contexto": estado["contexto"], "pregunta": estado["pregunta"]})
    respuesta = preservar_datos_clave(respuesta, estado["contexto"])
    urls = list(dict.fromkeys(re.findall(r"\[Fuente: (https?://[^\]]+)\]", estado["contexto"])))
    return {"respuesta": respuesta + "\n\nFuentes: " + ", ".join(urls) +
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
