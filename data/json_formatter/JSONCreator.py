import json
import re

def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", texto, flags=re.DOTALL)
    texto = re.sub(r"<image>", "", texto)
    texto = re.sub(r"\n+", "\n", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto.strip()

def unir_jsons_con_formato(path_gt, path_base, path_finetuned, output_path):
    with open(path_gt, "r", encoding="utf-8") as f:
        ref_list = json.load(f)
    with open(path_base, "r", encoding="utf-8") as f:
        base_list = json.load(f)
    with open(path_finetuned, "r", encoding="utf-8") as f:
        ft_list = json.load(f)

    # Diccionarios clave: imagen
    ref_dict = {item["image"]: item for item in ref_list}
    base_dict = {item["image"]: item["base_output"] for item in base_list}
    ft_dict = {item["image"]: item["finetuned_output"] for item in ft_list}

    imagenes_comunes = set(ref_dict) & set(base_dict) & set(ft_dict)
    resultado = []

    for img in sorted(imagenes_comunes):
        ref_item = ref_dict[img]
        input_text = limpiar_texto(ref_item.get("input_text", ""))
        output_text = limpiar_texto(ref_item.get("output_text", ""))
        base_output = limpiar_texto(base_dict[img])
        finetuned_output = limpiar_texto(ft_dict[img])

        if len(output_text) < 10 or len(base_output) < 10 or len(finetuned_output) < 10:
            continue  # descartar truncados

        resultado.append({
            "image": img,
            "input_text": input_text,
            "output_text": output_text,
            "base_output": base_output,
            "finetuned_output": finetuned_output
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"[✓] Unificado con éxito ({len(resultado)} muestras). Guardado en: {output_path}")
