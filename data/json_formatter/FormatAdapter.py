import json


class FormatAdapter:
    def __init__(self, json_path):
        self.json_path = json_path

    def load_dataset(self):
        """Carga el dataset desde el JSON original"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def adapt_entry(self, entrada):
        """Adapta una entrada al formato de fine-tuning"""
        return {
            "image": entrada["image"],
            "input_text": "<image>\nPregunta: ",
            "output_text": entrada["expected_output"]
        }

    def adapt_dataset(self, dataset):
        """Adapta todo el conjunto de datos"""
        return [self.adapt_entry(ej) for ej in dataset]

    def save_adapted_dataset(self, dataset, output_path):
        """Guarda el dataset adaptado"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Guardado {output_path} con {len(dataset)} ejemplos adaptados.")

    def run(self, output_path):
        """Ejecuta la adaptación completa"""
        dataset = self.load_dataset()
        adapted = self.adapt_dataset(dataset)
        self.save_adapted_dataset(adapted, output_path)