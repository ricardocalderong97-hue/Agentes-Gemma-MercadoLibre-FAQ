"""Evaluación reproducible del RAG FAQ Mercado Libre con LangSmith.

Distingue una respuesta fundamentada de una abstención y evita falsos positivos
por coincidencia parcial de palabras (por ejemplo 'no' en 'No tengo información').

Las reglas son específicas de las cuatro preguntas de este dataset de prueba;
no miden precisión semántica general ni verifican las políticas actuales online.
"""
import os
import re
import unicodedata

from langsmith import Client
from agentes import consultar_rag

# Dataset NUEVO: deja intactos los experimentos de la versión anterior.
DATASET = "tutorial-gemma-mercadolibre-faq-v4"
EJEMPLOS = [
    {
        "inputs": {"pregunta": "¿Puedo pagar contra entrega?"},
        "reference_outputs": {"caso": "contra_entrega"},
    },
    {
        "inputs": {"pregunta": "¿Cuánto tiempo tengo para pagar en Efecty?"},
        "reference_outputs": {"caso": "efecty"},
    },
    {
        "inputs": {"pregunta": "¿Puedo pagar usando Nequi o Daviplata?"},
        "reference_outputs": {"caso": "nequi_daviplata"},
    },
    {
        "inputs": {"pregunta": "¿Cuál es el salario del gerente de Mercado Libre?"},
        "reference_outputs": {"caso": "sin_evidencia"},
    },
]


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto).strip()


def es_abstencion(texto: str) -> bool:
    """Identifica expresiones de falta de información, no negaciones válidas."""
    t = normalizar(texto)
    patrones = [
        r"\bno (?:tengo|tenemos|hay|dispongo de|dispone de|cuento con) (?:esa |suficiente )?informacion\b",
        r"\bno (?:se|esta) (?:menciona|especifica|indica|disponible|en las faq)\b",
        r"\b(?:informacion|evidencia) insuficiente\b",
        r"\bno puedo (?:determinar|confirmar|responder)\b",
        r"\bno (?:lo|se) se\b",
    ]
    return any(re.search(p, t) for p in patrones)


def calificar_respuesta(caso: str, respuesta: str) -> tuple[bool, str]:
    """Reglas explícitas y auditables para este conjunto pequeño de pruebas."""
    t = normalizar(respuesta)
    if caso == "sin_evidencia":
        ok = es_abstencion(respuesta)
        return ok, "Se abstiene al faltar evidencia" if ok else "No reconoce que falta evidencia"

    if es_abstencion(respuesta):
        return False, "Se abstuvo pese a que el dato aparece en la FAQ local"

    if caso == "contra_entrega":
        # Exige una negación referente EXPLÍCITAMENTE al pago contra entrega.
        patrones = [
            r"\bno (?:se )?(?:ofrece|permite|acepta|admite|puede|esta disponible)\b.{0,85}\b(?:pago )?contra entrega\b",
            r"\b(?:pago )?contra entrega\b.{0,85}\bno (?:se )?(?:ofrece|permite|acepta|admite|esta disponible|es posible)\b",
            r"\b(?:pago )?contra entrega\b.{0,45}\bno\b.{0,30}\b(?:disponible|permitido|aceptado|ofrecido)\b",
            r"\bno\b.{0,25}\b(?:pagar|pagos?)\b.{0,25}\bcontra entrega\b",
        ]
        ok = any(re.search(p, t) for p in patrones)
        return ok, "Negación explícita del pago contra entrega" if ok else "No afirma claramente la política de pago contra entrega"

    if caso == "efecty":
        ok = bool(re.search(r"\b7\s*dias?\b|\bsiete\s*dias?\b", t)) and "efecty" in t
        return ok, "Indica Efecty y 7 días" if ok else "Falta Efecty o el plazo de 7 días"

    if caso == "nequi_daviplata":
        ok = "pse" in t and ("nequi" in t or "daviplata" in t)
        return ok, "Relaciona PSE con Nequi o Daviplata" if ok else "Falta la relación entre PSE y las billeteras"

    return False, f"Caso de evaluación desconocido: {caso}"


def target(inputs: dict) -> dict:
    return {"respuesta": consultar_rag(inputs["pregunta"])["respuesta"]}


def respuesta_faq_correcta(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    caso = reference_outputs["caso"]
    ok, razon = calificar_respuesta(caso, outputs.get("respuesta", ""))
    return {"key": "respuesta_faq_correcta", "score": int(ok), "comment": razon}


def main():
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Configura LANGSMITH_API_KEY en .env para evaluar.")
    client = Client()
    # IMPORTANTE: un dataset puede existir aunque este vacio (p. ej. una
    # ejecucion anterior que creo el dataset pero no cargo los ejemplos).
    if not client.has_dataset(dataset_name=DATASET):
        client.create_dataset(
            dataset_name=DATASET,
            description="FAQ Mercado Libre: pago contra entrega, Efecty, PSE y abstencion sin evidencia. Evaluacion v3.",
        )

    existentes = list(client.list_examples(dataset_name=DATASET))
    preguntas_existentes = {
        ejemplo.inputs.get("pregunta")
        for ejemplo in existentes
        if ejemplo.inputs
    }
    faltantes = [
        ejemplo for ejemplo in EJEMPLOS
        if ejemplo["inputs"]["pregunta"] not in preguntas_existentes
    ]
    if faltantes:
        print(f"Agregando {len(faltantes)} ejemplos faltantes al dataset '{DATASET}'...")
        # LangSmith requiere la clave reference_outputs para asociar las
        # etiquetas de referencia a cada ejemplo.
        client.create_examples(dataset_name=DATASET, examples=faltantes)

    # Confirmar antes de llamar a evaluate; si el dataset esta vacio,
    # LangSmith lanza StopIteration con un mensaje poco claro.
    import time
    ejemplos = []
    for _ in range(5):
        ejemplos = list(client.list_examples(dataset_name=DATASET))
        if len(ejemplos) >= len(EJEMPLOS):
            break
        time.sleep(1)
    print(f"Ejemplos disponibles para evaluar: {len(ejemplos)}")
    if not ejemplos:
        raise RuntimeError("El dataset sigue vacio en LangSmith. Comprueba conexion/permisos e intenta nuevamente.")

    resultados = client.evaluate(
        target,
        data=DATASET,
        evaluators=[respuesta_faq_correcta],
        experiment_prefix="gemma3-1b-bm25-faq-ml-v4",
        max_concurrency=1,
    )
    df = resultados.to_pandas()
    columnas = [c for c in ["inputs.pregunta", "outputs.respuesta", "feedback.respuesta_faq_correcta", "execution_time"] if c in df.columns]
    print("\n=== EVALUACIÓN FAQ V4 ===")
    print(df[columnas].to_string(index=False))
    scores = df.get("feedback.respuesta_faq_correcta")
    if scores is not None:
        validos = scores.dropna()
        if len(validos):
            print(f"\nCoincidencias con criterios específicos: {int(validos.sum())}/{len(validos)} ({validos.mean():.0%})")
    print("Nota: evaluación limitada a estos casos; no mide la calidad general del RAG.")


if __name__ == "__main__":
    main()
