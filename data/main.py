from data.json_formatter.FormatAdapter import FormatAdapter
from data.json_formatter.SelectImages import SelectImages


def prepare_dataset(input_path, filtered_path):
    # 1. Filtrar imágenes
    selector = SelectImages(input_path)
    selector.run(filtered_path)

    # 2. Adaptar formato para fine-tuning
    adapter = FormatAdapter(filtered_path)
    adapter.run(filter_output)

if __name__ == "__main__":
    input_json = r"C:\Users\aaron\Desktop\Final\base_qwen_inference.json"
    filter_output = r"C:\Users\aaron\Desktop\Examenes\JSON_fine_tuning\subset_finetune_cien.json"

    prepare_dataset(input_json, filter_output)
