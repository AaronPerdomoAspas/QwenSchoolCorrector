import json

from src.evaluation.gpt_judge.GPTPrompts import evaluate_with_g_eval

INPUT_JSON = "resultados_completos.json"
OUTPUT_JSON = "g_eval_resultados_base.json"  # o _finetuned si quieres evaluar esa

# === CARGA DE DATOS ===
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# === APLICAR G-EVAL ===
evaluaciones = []

for i, sample in enumerate(data):
    print(f"🧪 Evaluando muestra {i+1}/{len(data)}...")
    resultados = evaluate_with_g_eval(
        question=sample["input_text"].replace("<image>", "[imagen]"),
        context="(sin contexto visual)",
        answer=sample["base_output"]
    )

    sample_result = {
        "image": sample["image"],
        "coherence": resultados["coherence"],
        "consistency": resultados["consistency"],
        "fluency": resultados["fluency"],
        "relevance": resultados["relevance"]
    }

    evaluaciones.append(sample_result)

# === GUARDADO DE RESULTADOS ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(evaluaciones, f, indent=2, ensure_ascii=False)

print(f"✅ Evaluaciones G-Eval guardadas en {OUTPUT_JSON}")
