# Agentes con Gemma 3:1b, LangChain y RAG de Mercado Libre

Proyecto académico desarrollado en **Python y Visual Studio Code** con un modelo de lenguaje local (**Gemma 3:1b**, a través de **Ollama**). Incluye herramientas de cálculo y clima, un RAG de preguntas frecuentes de Mercado Libre Colombia, un traductor, memoria conversacional y evaluación de respuestas en **LangSmith**.

> **Este proyecto no es un servicio oficial de Mercado Libre.** La base FAQ es una copia local resumida de información pública y **no se actualiza automáticamente**. Para comprobar condiciones vigentes, consulta los enlaces oficiales citados en las respuestas.

## 1. ¿Qué contiene?

| Componente | Descripción | Comando |
|---|---|---|
| Calculadora y clima | Suma, resta, multiplicación, división, potencias, paréntesis y clima simulado (o real opcional) | `python main.py herramientas --pregunta "¿Cuánto es (8+4)*3?"` |
| RAG FAQ | Preguntas frecuentes de Mercado Libre Colombia recuperadas con BM25 y reordenamiento | `python main.py rag --pregunta "¿Puedo pagar contra entrega?"` |
| Traductor | Traduce texto mediante Gemma local | `python main.py traducir --texto "Mi compra llegó ayer" --origen español --destino inglés` |
| Memoria | Ejemplo de conversación con historial en memoria durante la ejecución | `python main.py memoria` |
| Evaluación | Cuatro casos de prueba en un dataset de LangSmith | `python evaluar.py` |

**Arquitectura:** LangChain conecta el modelo con los prompts y herramientas; LangGraph organiza el flujo de ejecución; Ollama ejecuta Gemma en el computador; LangSmith permite observar las trazas y comparar evaluaciones. La selección de herramientas del primer flujo se hace mediante reglas de Python, por lo que **no es un agente con selección autónoma de herramientas**.

## 2. Requisitos previos (Windows)

Instala en el computador:

1. **Python 3.12 o 3.14**: <https://www.python.org/downloads/>. El proyecto se ha probado con Python 3.14.0.
2. **Visual Studio Code**: <https://code.visualstudio.com/>. Se recomienda instalar las extensiones **Python** y **Jupyter**.
3. **Ollama**: <https://ollama.com/download>.
4. **Git**, solamente si vas a descargar mediante `git clone`: <https://git-scm.com/downloads/win>.
5. Conexión a Internet **para descargar dependencias y el modelo**. Después, las herramientas locales pueden funcionar sin Internet. LangSmith y el clima real sí necesitan conexión.

Para confirmar Python en **PowerShell** (Terminal → New Terminal en VS Code):

```powershell
py --list
```

Si aparece `3.14`, usa `py -3.14` en las instrucciones siguientes. Si aparece `3.12` pero no `3.14`, reemplaza `py -3.14` por `py -3.12`. Si el comando `python` no funciona antes de crear el entorno virtual, pero `py --list` sí muestra Python, **no es un problema**: sigue usando `py` para crear `.venv`.

## 3. Descarga del proyecto

**Opción A: clonar GitHub** (requiere Git). Abre PowerShell en la carpeta donde quieras guardar el proyecto y pega:

```powershell
git clone https://github.com/ricardocalderong97-hue/Agentes-Gemma-MercadoLibre-FAQ.git
cd Agentes-Gemma-MercadoLibre-FAQ
code .
```

Si `code .` no es reconocido, abre VS Code manualmente y selecciona **File → Open Folder** para escoger la carpeta descargada.

**Opción B: sin Git.** Abre el [repositorio](https://github.com/ricardocalderong97-hue/Agentes-Gemma-MercadoLibre-FAQ), selecciona **Code → Download ZIP**, descomprime **todo el ZIP**, y en VS Code selecciona **File → Open Folder**. La carpeta abierta debe contener `main.py`, `requirements.txt` y `.env.example`. **No ejecutes scripts desde dentro del ZIP sin extraerlo.**

> Todos los comandos siguientes se ejecutan en la **raíz del proyecto** (la carpeta que contiene `main.py`).

## 4. Instalación manual, paso a paso (PowerShell)

### Paso 1: crear el entorno virtual

En la terminal de VS Code:

```powershell
py -3.14 -m venv .venv
```

Para Python 3.12 utiliza `py -3.12 -m venv .venv`.

### Paso 2: activar el entorno virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

La terminal debe comenzar con `(.venv)`. Si PowerShell indica que la ejecución de scripts está bloqueada, ejecuta:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Este permiso se aplica **solo a la ventana actual** de PowerShell. Comprueba la versión:

```powershell
python --version
```

### Paso 3: instalar las dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para comprobar que se instalaron las principales bibliotecas:

```powershell
python -m pip show langchain langgraph langchain-ollama langsmith pandas
```

**No copies una carpeta `.venv` de otro computador.** Debes crearla con el Python instalado en el nuevo equipo.

### Paso 4: comprobar Ollama y descargar Gemma

Asegúrate de que la aplicación **Ollama** esté abierta. Ejecuta:

```powershell
ollama list
```

Si `gemma3:1b` **no aparece** en la lista, descárgalo (aproximadamente 800 MB):

```powershell
ollama pull gemma3:1b
```

Si **ya aparece**, **omite `ollama pull`**: el modelo está disponible en ese computador.

Prueba el modelo:

```powershell
ollama run gemma3:1b "Hola, responde en español."
```

### Paso 5: crear la configuración local `.env`

En la raíz del proyecto:

```powershell
Copy-Item .env.example .env
```

Abre `.env` en VS Code y verifica estas variables:

```dotenv
OLLAMA_MODEL=gemma3:1b
OLLAMA_BASE_URL=http://localhost:11434
CLIMA_REAL=false

LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=agentes-gemma-mercadolibre
```

Con `LANGSMITH_TRACING=false` puedes probar los agentes sin registrarte en LangSmith. `CLIMA_REAL=false` conserva el clima **simulado** para una demostración independiente de la API meteorológica. **No compartas ni subas a GitHub tu archivo `.env`.** Está excluido mediante `.gitignore`.

### Paso 6: ejecutar la demostración

```powershell
python main.py demo
```

Debes ver tres apartados: **agente de herramientas**, **RAG** y **traductor**. Si aparecen errores de conexión, comprueba primero `ollama list` y que la aplicación Ollama esté abierta.

## 5. Cómo hacer preguntas a los agentes

**Calculadora** (se puede cambiar la pregunta):

```powershell
python main.py herramientas --pregunta "¿Cuánto es 15 más 8?"
python main.py herramientas --pregunta "¿Cuánto es 144 dividido entre 12?"
python main.py herramientas --pregunta "¿Cuánto es 2 elevado a 5?"
python main.py herramientas --pregunta "¿Cuánto es (8+4)*3?"
```

**Clima y cálculo en una misma pregunta:**

```powershell
python main.py herramientas --pregunta "¿Qué clima hace en Cali y cuánto es 18 dividido entre 3?"
```

Con `CLIMA_REAL=false`, el clima **no es una observación actual**. Para consultar Open-Meteo en tiempo real, cambia a `CLIMA_REAL=true` y asegúrate de tener Internet y DNS funcionando.

**RAG de Mercado Libre Colombia:**

```powershell
python main.py rag --pregunta "¿Puedo pagar contra entrega?"
python main.py rag --pregunta "¿Puedo pagar con Nequi o Daviplata?"
python main.py rag --pregunta "¿Cuánto tiempo tengo para pagar en Efecty?"
python main.py rag --pregunta "¿Cuál es el salario del gerente de Mercado Libre?"
```

Para la última pregunta, la respuesta correcta es indicar que **no hay información suficiente en la base FAQ**. La base local está en `datos/faq_mercadolibre_colombia.json`. Las respuestas incluyen las fuentes que aparecen en ese archivo.

**Traductor:**

```powershell
python main.py traducir --texto "Mi compra llegará mañana" --origen español --destino inglés
python main.py traducir --texto "The package has arrived" --origen inglés --destino español
```

**Memoria conversacional (ejemplo):**

```powershell
python main.py memoria
```

La memoria del ejemplo se conserva **durante la ejecución**, no indefinidamente entre reinicios.

Cada comando hace una consulta y termina. No se incluye un chat conversacional permanente por terminal.

## 6. Activar LangSmith (opcional, recomendado para evaluar)

1. Crea una cuenta en <https://smith.langchain.com/>.
2. En **Settings → API Keys**, crea una API key. Copia la clave al crearla.
3. Abre el archivo `.env` y configura:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=TU_CLAVE_PRIVADA
LANGSMITH_PROJECT=agentes-gemma-mercadolibre
```

4. Guarda `.env` y vuelve a ejecutar un agente, por ejemplo:

```powershell
python main.py rag --pregunta "¿Puedo pagar contra entrega?"
```

5. En LangSmith busca el proyecto `agentes-gemma-mercadolibre` y abre la sección **Tracing**. Allí puedes observar las entradas, respuestas y, cuando están instrumentados, los pasos de recuperación y la llamada a Ollama.

> Gemma sigue ejecutándose localmente, pero **las trazas enviadas a LangSmith pueden contener tus preguntas y las respuestas del modelo**. No registres información confidencial ni compartas la API key. El profesor puede usar **su propia cuenta** y su propia clave.

## 7. Ejecutar la evaluación del RAG

Con LangSmith configurado y conexión a Internet:

```powershell
python evaluar.py
```

Este script crea o reutiliza el dataset `tutorial-gemma-mercadolibre-faq-v4`, verifica que tenga cuatro preguntas y lanza una evaluación específica de las respuestas. La primera ejecución puede crear los ejemplos; las siguientes reutilizan el dataset. Las evaluaciones aparecen en **Datasets & Experiments** de LangSmith y los resultados se imprimen en PowerShell.

Las cuatro pruebas revisan:

| Caso | Criterio de evaluación |
|---|---|
| Pago contra entrega | Negación explícita de esa modalidad, respaldada por la FAQ local |
| Efecty | Mención de Efecty y plazo de siete días |
| Nequi o Daviplata | Relación de estas opciones con PSE |
| Salario del gerente | Abstención por falta de información documental |

En una ejecución de referencia de esta versión se cumplieron los **4 de 4 criterios (100 %)**. Esta cifra **no** demuestra exactitud general del RAG: son solamente cuatro casos específicos. Algunas respuestas pueden contener una aclaración adicional respaldada por el documento cuando Gemma omite un detalle relevante.

## 8. Notebook de clase (opcional)

Abre `notebook_puntos_6_al_10.ipynb` en VS Code, con la extensión Jupyter instalada. En la esquina superior derecha selecciona el kernel asociado a `.venv` y ejecuta las celdas por orden. El notebook acompaña a los scripts Python: para la versión más reciente de la lógica de RAG utiliza `agentes.py` y `evaluar.py`.

## 9. Instalación automática (alternativa)

Para quienes prefieran evitar los pasos de PowerShell, también se incluyen **`INSTALAR.bat`** e **`INICIAR.bat`**. Extrae el proyecto antes de usarlos. `INSTALAR.bat` intenta detectar Python, instalar dependencias, comprobar Ollama y descargar Gemma **solo si hace falta**. Si hay restricciones de permisos, DNS o Internet, utiliza la instalación manual anterior.

Cuando termine la instalación, ejecuta `INICIAR.bat` para usar un menú de opciones de los agentes.

## 10. Problemas frecuentes

| Problema | Cómo solucionarlo |
|---|---|
| `No suitable Python runtime found` | Ejecuta `py --list` y usa una versión detectada (`py -3.14` o `py -3.12`). Instala Python si no aparece ninguna. |
| `python` no se reconoce | Crea el entorno con `py`, luego activa `.venv`; dentro del entorno, `python` debe funcionar. |
| PowerShell bloquea `Activate.ps1` | Ejecuta `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelve a activar el entorno. |
| `ModuleNotFoundError` | Confirma que aparece `(.venv)` e instala `python -m pip install -r requirements.txt`. |
| No se conecta a Ollama | Abre Ollama y prueba `ollama list` o `ollama run gemma3:1b`. |
| `ollama pull` falla por DNS | Si `ollama list` ya muestra `gemma3:1b`, no necesitas descargarlo otra vez. |
| El clima real falla | Utiliza `CLIMA_REAL=false`; para modo real comprueba acceso a `geocoding-api.open-meteo.com`. |
| No aparecen trazas en LangSmith | Comprueba `LANGSMITH_TRACING=true`, API key correcta, nombre del proyecto y conexión a Internet. |
| La evaluación no muestra resultados | Ejecuta `python evaluar.py`, revisa el mensaje completo y confirma que `pandas` esté instalado. |
| Ejecuto scripts dentro del ZIP y fallan | Extrae el ZIP completamente a una carpeta normal y vuelve a abrirla en VS Code. |

## 11. Volver a abrir el proyecto otro día

1. Abre VS Code → **File → Open Folder** → selecciona la carpeta del proyecto.
2. Abre **Terminal → New Terminal**.
3. Activa el entorno y prueba el agente que necesites:

```powershell
.\.venv\Scripts\Activate.ps1
ollama list
python main.py rag --pregunta "¿Cómo puedo devolver una compra?"
```

**No tienes que reinstalar Python, las dependencias ni Gemma cada vez que abras VS Code.**

## 12. Archivos principales y fuentes

- `main.py`: comandos de ejecución.
- `agentes.py`: lógica de calculadora, clima, recuperación de FAQ y traducción.
- `herramientas.py`: funciones de cálculo y clima.
- `datos/faq_mercadolibre_colombia.json`: preguntas y respuestas locales con URLs de referencia.
- `evaluar.py`: casos y criterios de validación en LangSmith.
- `memoria.py`: ejemplo de memoria conversacional.
- `.env.example`: plantilla de configuración sin credenciales.
- `requirements.txt`: dependencias Python.
- `INSTALAR.bat`, `INICIAR.bat`: instalador y menú opcionales.

Páginas oficiales citadas en la base de ejemplo: [Medios de pago](https://www.mercadolibre.com.co/l/mediosdepago), [Compra Protegida](https://www.mercadolibre.com.co/compra-protegida/), [Compra segura](https://www.mercadolibre.com.co/l/compra-seguro) y [Agencias](https://envios.mercadolibre.com.co/agencias).

---

**Alcance y limitaciones:** Es una demostración académica; no es un asistente oficial de Mercado Libre ni representa información actualizada de las políticas comerciales. No requiere APIs de OpenAI ni Anthropic. LangSmith es opcional para el uso básico, pero necesario para reproducir la evaluación y las trazas de clase.
