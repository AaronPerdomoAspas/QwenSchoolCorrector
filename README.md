# 🧠 Evaluador Multimodal de Actividades Escolares

[![My Skills](https://skillicons.dev/icons?i=python,pytorch,jupyter)](https://skillicons.dev)
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub Logo" width="40" style="margin-left: 5px">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/anaconda/anaconda-original.svg" alt="Anaconda Logo" width="45" style="margin-left: 5px">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" alt="VSCode Logo" width="40" style="margin-left: 5px">

---

## 📑 Tabla de Contenidos

- [Introducción](#introducción)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Ejecución del Sistema](#ejecución-del-sistema)
- [Evaluación del Modelo](#evaluación-del-modelo)
- [Subida a Hugging Face](#subida-a-hugging-face)
- [Créditos Académicos](#créditos-académicos)

---

## 📘 Introducción

Este repositorio forma parte del Trabajo de Fin de Grado en Ciencia e Ingeniería de Datos llamado "Desarrollo de un sistema de evaluación de actividades escolares
mediante modelos de visión y de procesamiento del lenguaje natural". El proyecto emplea el modelo **Qwen2.5-VL-7B-Instruct**, realizando fine-tuning supervisado, evaluación automática y validación mediante LLMs externos.

---

## 🧱 Estructura del Proyecto

```
data/
  huggingface_uploader/          # Subida del modelo fine-tuneado a la plataforma
  json_formatter/                # Preparación del conjunto de datos
notebooks/
  qwen_inference.ipynb           # Inferencia con modelo base
  qwen_finetuning.ipynb          # Fine-tuning supervisado con LoRA
src/
  evaluation/
    gpt_judge/
        g_evaluation.ipynb           # Evaluación con GPT-as-a-judge
        GPTPrompts.py                # Prompts de evaluación
    gpt_judge_representation         # Visualización avanzada
    metrics/
        AutomaticEvaluator.py        # Métricas tradicionales (BLEU, ROUGE...)
        MetricsRunner.py             # Comparación base vs fine-tuned
    metrics_representation.ipynb
```

---

## ⚙️ Requisitos e Instalación

Se recomienda crear un entorno virtual con Python 3.11+.  
Instalación de dependencias:

```bash
pip install -r requirements.txt
```

Librerías clave: `transformers`, `torch`, `peft`, `openai`, `datasets`, `evaluate`, `bert_score`.

---

## ▶️ Ejecución del Sistema

- **Carga del modelo e inferencia con el modelo base:**
  ```
  jupyter notebook notebooks/qwen_inference.ipynb
  ```

- **Fine-tuning supervisado:**
  ```
  jupyter notebook notebooks/qwen_finetuning.ipynb
  ```

- **Evaluación automática (BLEU, ROUGE, etc.):**
  ```
  jupyter notebook src/evaluation/metrics_representation.ipynb
  ```

- **Evaluación GPT-as-a-judge (requiere API de OpenAI):**
  ```
  jupyter notebook src/evaluation/gpt_judge/g_evaluation.ipynb
  jupyter notebook src/evaluation/gpt_judge_representation.ipynb
  ```

---

## 📤 Subida a Hugging Face

El script `data/huggingface_uploader/HuggingFaceUploader.py` permite cargar los modelos entrenados a tu espacio personal en [Hugging Face](https://huggingface.co). Configura previamente tu token con:

```bash
huggingface-cli login
```

Podrás encontrar el modelo en HuggingFace: AaronPA/qwen2vl-finetuned-school-corrector.

---

## 🎓 Créditos Académicos

Este proyecto fue desarrollado como parte del **Trabajo de Fin de Grado** en Ciencia e Ingeniería de Datos – Universidad de Las Palmas de Gran Canaria.

**Autor**: Aarón Perdomo Aspas  
**Tutoras**: María Dolores Afonso Suárez y Victoria Torres Rodríguez  
Año académico: 2024/2025

---
