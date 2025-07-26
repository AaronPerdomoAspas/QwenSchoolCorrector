import os
from huggingface_hub import upload_folder, login

def upload_finetuned_model():
    # Autenticación
    token = "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"  #
    login(token=token)

    # Ruta al modelo fine-tuneado
    current_dir = os.path.dirname(__file__)
    folder_path = os.path.abspath(os.path.join(current_dir, "..", "..", "outputs_qwen2vl_finetuned", "checkpoint-100"))

    # Repositorio en Hugging Face
    repo_id = "AaronPA/qwen2vl-finetuned-school-corrector"

    # Subida del modelo completo
    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=".",
        allow_patterns="*"
    )

if __name__ == "__main__":
    upload_finetuned_model()
