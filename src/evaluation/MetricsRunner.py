import json
from tqdm import tqdm
from src.evaluation.AutomaticEvaluator import AutomaticEvaluator

class MetricsRunner:
    def __init__(self, base_path: str, finetuned_path: str, gold_path: str, lang: str = "es"):
        self.base_path = base_path
        self.finetuned_path = finetuned_path
        self.gold_path = gold_path
        self.lang = lang
        self.metric_engine = AutomaticEvaluator(lang=lang)

        self.base_data = self._load_json(self.base_path)
        self.finetuned_data = self._load_json(self.finetuned_path)
        self.gold_data = self._load_json(self.gold_path)

        assert len(self.base_data) == len(self.finetuned_data) == len(self.gold_data), \
            "Los archivos deben tener la misma cantidad de ejemplos."

    def _load_json(self, path: str) -> list:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self) -> list:
        results = []
        for i in tqdm(range(len(self.base_data)), desc="Evaluando pares"):
            base_pred = self.base_data[i]["base_output"]
            finetuned_pred = self.finetuned_data[i]["finetuned_output"]
            reference = self.gold_data[i]["output_text"]

            base_metrics = self.metric_engine.compute_all(base_pred, reference)
            finetuned_metrics = self.metric_engine.compute_all(finetuned_pred, reference)

            results.append({
                "id": i,
                "reference": reference,
                "base_answer": base_pred,
                "finetuned_answer": finetuned_pred,
                "base_metrics": base_metrics,
                "finetuned_metrics": finetuned_metrics
            })

        return results