"""Punto 9: evaluación reproducible del RAG con LangSmith, sin LLM como juez."""
import os
from langsmith import Client
from agentes import consultar_rag

DATASET = "tutorial-gemma-mercadolibre-faq-v2"
EJEMPLOS = [
    {"inputs": {"pregunta": "¿Puedo pagar contra entrega?"}, "outputs": {"respuesta_esperada": "no"}},
    {"inputs": {"pregunta": "¿Cuánto tiempo tengo para pagar en Efecty?"}, "outputs": {"respuesta_esperada": "7 días"}},
    {"inputs": {"pregunta": "¿Puedo pagar usando Nequi o Daviplata?"}, "outputs": {"respuesta_esperada": "PSE"}},
]



def target(inputs: dict) -> dict:
    return {"respuesta": consultar_rag(inputs["pregunta"])["respuesta"]}


def contiene_respuesta(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Métrica sencilla del notebook: no equivale a evaluación semántica exhaustiva."""
    return reference_outputs["respuesta_esperada"].lower() in outputs["respuesta"].lower()


def main():
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Crea una API key en LangSmith y configúrala en .env antes de evaluar.")
    client = Client()
    if not client.has_dataset(dataset_name=DATASET):
        dataset = client.create_dataset(dataset_name=DATASET, description="Evaluación de preguntas frecuentes públicas Mercado Libre Colombia")
        client.create_examples(dataset_id=dataset.id, examples=EJEMPLOS)
    resultados = client.evaluate(
        target,
        data=DATASET,
        evaluators=[contiene_respuesta],
        experiment_prefix="gemma3-1b-bm25-faq-ml",
        max_concurrency=1,
    )
    print(resultados.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
