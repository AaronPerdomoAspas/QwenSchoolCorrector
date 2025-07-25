# g_eval_metrics.py

from openai import OpenAI

client = OpenAI(api_key="tu_api_key_aquí")


# ===== PROMPTS =====

def build_prompt(metric: str, question: str, context: str, answer: str) -> str:
    if metric == "coherence":
        return f"""Evalúa la siguiente respuesta en la dimensión de **coherencia** (1-5). 
Una respuesta coherente está bien estructurada, fluye naturalmente y no presenta contradicciones.

Pregunta:
{question}

Contexto:
{context}

Respuesta generada:
{answer}

Solo responde con un número entre 1 y 5:"""

    elif metric == "consistency":
        return f"""Evalúa la siguiente respuesta en la dimensión de **consistencia** (1-5). 
Debe ser fiel a la información del contexto, sin inventos ni contradicciones.

Pregunta:
{question}

Contexto:
{context}

Respuesta generada:
{answer}

Solo responde con un número entre 1 y 5:"""

    elif metric == "fluency":
        return f"""Evalúa la siguiente respuesta en la dimensión de **fluidez** (1-3). 
Debe estar bien escrita, sin errores gramaticales ni incoherencias formales.

Respuesta generada:
{answer}

Solo responde con un número entre 1 y 3:"""

    elif metric == "relevance":
        return f"""Evalúa la siguiente respuesta en la dimensión de **relevancia** (1-5). 
Debe responder directamente a la pregunta y utilizar la información más importante del contexto.

Pregunta:
{question}

Contexto:
{context}

Respuesta generada:
{answer}

Solo responde con un número entre 1 y 5:"""

    else:
        raise ValueError("Métrica desconocida.")


# ===== FUNCIÓN PRINCIPAL =====

def evaluate_with_g_eval(question: str, context: str, answer: str) -> dict:
    metrics = ["coherence", "consistency", "fluency", "relevance"]
    resultados = {}

    for metric in metrics:
        prompt = build_prompt(metric, question, context, answer)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            value = float(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error evaluando {metric}: {e}")
            value = None
        resultados[metric] = value

    return resultados
