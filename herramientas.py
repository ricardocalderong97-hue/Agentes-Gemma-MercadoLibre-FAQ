"""Punto 6: herramientas del agente, como en el notebook original."""
import os
import requests
from langchain.tools import tool


@tool
def multiplicar(a: float, b: float) -> float:
    """Multiplica exactamente dos números dados por el usuario."""
    return a * b


import ast
import math


@tool
def calcular(expresion: str) -> str:
    """Calcula de forma segura suma, resta, multiplicación, división y potencias.

    Admite expresiones numéricas como 12+8, 10/2, 2**3, (3+4)*2.
    No utiliza eval() ni ejecuta código arbitrario.
    """
    operaciones = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** b,
    }

    def evaluar(nodo, profundidad=0):
        if profundidad > 12:
            raise ValueError("La expresión es demasiado compleja")
        if isinstance(nodo, ast.Expression):
            return evaluar(nodo.body, profundidad + 1)
        if isinstance(nodo, ast.Constant) and type(nodo.value) in (int, float):
            return nodo.value
        if isinstance(nodo, ast.UnaryOp) and isinstance(nodo.op, (ast.UAdd, ast.USub)):
            valor = evaluar(nodo.operand, profundidad + 1)
            return valor if isinstance(nodo.op, ast.UAdd) else -valor
        if isinstance(nodo, ast.BinOp) and type(nodo.op) in operaciones:
            a = evaluar(nodo.left, profundidad + 1)
            b = evaluar(nodo.right, profundidad + 1)
            if isinstance(nodo.op, ast.Div) and b == 0:
                raise ValueError("No se puede dividir entre cero")
            if isinstance(nodo.op, ast.Pow) and (abs(b) > 12 or abs(a) > 1000000):
                raise ValueError("Potencia demasiado grande")
            resultado = operaciones[type(nodo.op)](a, b)
            if isinstance(resultado, complex) or not math.isfinite(resultado) or abs(resultado) > 1e12:
                raise ValueError("Resultado fuera de rango")
            return resultado
        raise ValueError("Solo se admiten números y operadores +, -, *, /, ** y paréntesis")

    if len(expresion) > 120:
        return "Error: expresión demasiado larga."
    try:
        arbol = ast.parse(expresion, mode="eval")
        valor = evaluar(arbol)
        return f"{valor:g}" if isinstance(valor, (int, float)) else str(valor)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Error de cálculo: {exc}"


@tool
def clima(ciudad: str) -> str:
    """Consulta el clima de una ciudad. Por defecto usa el clima FICTICIO del tutorial.

    Para usar datos actuales, define CLIMA_REAL=true en el archivo .env.
    """
    if os.getenv("CLIMA_REAL", "false").lower() != "true":
        return f"Clima SIMULADO del ejercicio para {ciudad}: 24 °C y soleado. No son datos actuales."

    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": ciudad, "count": 1, "language": "es", "format": "json"},
            timeout=15,
        )
        geo.raise_for_status()
        lugares = geo.json().get("results", [])
        if not lugares:
            return f"No encontré la ciudad '{ciudad}' en Open-Meteo."
        lugar = lugares[0]
        meteo = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lugar["latitude"],
                "longitude": lugar["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=15,
        )
        meteo.raise_for_status()
        actual = meteo.json()["current"]
        return (
            f"Clima ACTUAL según Open-Meteo en {lugar['name']}, {lugar.get('country', '')}: "
            f"{actual['temperature_2m']} °C, humedad {actual['relative_humidity_2m']} %, "
            f"viento {actual['wind_speed_10m']} km/h, código meteorológico WMO "
            f"{actual['weather_code']}. Hora local de medición: {actual['time']}."
        )
    except (requests.RequestException, KeyError, ValueError) as exc:
        return f"No fue posible obtener datos actuales de clima: {exc}"
