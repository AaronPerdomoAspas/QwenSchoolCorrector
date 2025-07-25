import json
import os


class SelectImages:
    def __init__(self, json_path):
        self.json_path = json_path
        self.selected_images = self._get_selected_image_names()

    def _get_selected_image_names(self):
        """Devuelve la lista de imágenes seleccionadas para el fine-tuning"""
        return [
            f"img{i}.jpg" for i in range(201, 230)
        ]

    def load_json_data(self):
        """Carga el archivo JSON original"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def filter_images(self, data):
        """Filtra los ejemplos cuya imagen está en la lista seleccionada"""
        return [ej for ej in data if ej["image"] in self.selected_images]

    def save_filtered_data(self, filtered_data, output_path):
        """Guarda los ejemplos filtrados en un nuevo archivo JSON"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        print(f"Guardado {output_path} con {len(filtered_data)} ejemplos.")

    def run(self, output_path):
        """Ejecución completa: carga, filtra y guarda"""
        data = self.load_json_data()
        filtered = self.filter_images(data)
        self.save_filtered_data(filtered, output_path)