# Agentes LangChain + Gemma local — versión actualizada

Esta versión mantiene **Gemma 3:1b en Ollama**, LangChain, LangGraph y trazabilidad de LangSmith. 

## Cambios principales

- **Punto 6**: clima simulado (`CLIMA_REAL=false`) o real cuando haya Internet, y calculadora con suma, resta, multiplicación, división, potencias, paréntesis y operaciones combinadas. Los cálculos se ejecutan mediante una herramienta Python segura, no por el LLM.
- **Punto 7**: RAG BM25 sobre una base **local** de preguntas frecuentes de **Mercado Libre Colombia**, con URLs oficiales visibles. Esta es una selección resumida de páginas públicas, no un asistente oficial ni una consulta web en vivo.
- **Punto 8**: traductor español/inglés y otros idiomas, sin cambios.
- **Punto 9**: evaluación del RAG actualizada a la nueva FAQ en LangSmith.
- **Punto 10**: memoria conversacional, sin cambios.

## Instalación en Windows

1. Extraer el ZIP y ejecutar `INSTALAR.bat` (o reutilizar el entorno virtual del proyecto anterior).
2. Configurar `.env` desde `.env.example` y mantener `CLIMA_REAL=false` para una demo sin red.
3. Ejecutar `INICIAR.bat`, o desde VS Code activar `.venv` y usar los comandos:

```powershell
python main.py herramientas --pregunta "¿Cuánto es 8+4? ¿Y 2^5?"
python main.py herramientas --pregunta "¿Qué clima hace en Cali y cuánto es 20/4?"
python main.py rag --pregunta "¿Puedo pagar con Nequi o Daviplata?"
python main.py rag --pregunta "¿Cuántos días tengo para devolver una compra?"
python main.py traducir --texto "Mi compra llegó ayer" --origen español --destino inglés
python main.py demo
python evaluar.py
```

## Consideraciones

- No se requieren claves de OpenAI o Anthropic.
- Para registrar trazas en LangSmith hay que configurar en `.env` `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...` y `LANGSMITH_PROJECT=...` (no compartir `.env`).
- La consulta del clima real requiere acceso a Open-Meteo; el modo simulado funciona sin conexión.
- La FAQ fue preparada a partir de páginas oficiales consultadas el **18 de noviembre de 2026** y puede quedar desactualizada. Las páginas originales están referenciadas en `datos/faq_mercadolibre_colombia.json`.
- El sistema es una **orquestación de herramientas y RAG**, no un agente con decisión autónoma de llamadas a herramientas.
- Para instalar una nueva copia, nunca copies `.venv` ni el archivo privado `.env`.

## Preguntas de ejemplo FAQ

- ¿Cómo devuelvo un producto?
- ¿Puedo pagar con PayPal?
- ¿Qué pasa si mi compra no llega?
- ¿Cuánto tiempo tengo para pagar por Efecty?
- ¿Puedo pagar contra entrega?

## Fuentes oficiales

- [Compra Protegida](https://www.mercadolibre.com.co/compra-protegida/)
- [Medios de pago](https://www.mercadolibre.com.co/l/mediosdepago)
- [Compra segura](https://www.mercadolibre.com.co/l/compra-seguro)
- [Agencias](https://envios.mercadolibre.com.co/agencias)

