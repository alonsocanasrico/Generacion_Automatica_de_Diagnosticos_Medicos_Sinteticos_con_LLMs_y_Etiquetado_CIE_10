# Generacion_Automatica_de_Diagnosticos_Medicos_Sinteticos_con_LLMs_y_Etiquetado_CIE_10

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Máster (TFM) titulado:

**"Generación Automática de Diagnósticos Médicos Sintéticos con LLMs y Etiquetado CIE-10"**  
Autor: Alonso Cañas Rico  
Máster en Aprendizaje Automático y Datos Masivos  
Universidad Politécnica de Madrid, 2025

## Resumen del proyecto

El objetivo de este trabajo es desarrollar un sistema automatizado para la generación de diagnósticos médicos sintéticos etiquetados con códigos CIE-10, utilizando modelos de lenguaje grandes (LLMs) como GPT-4o-mini (OpenAI) y LLaMA-3.2-3B-Instruct (local).

El sistema incluye:

- **Pipeline de generación** parametrizable (modelo, temperatura, penalizaciones, número de ejemplos, etc.).
- **Soporte multientorno**: funciona tanto con modelos en la nube como locales.
- **Evaluación de calidad** de los textos generados en varias dimensiones:
  - Similitud semántica (embeddings).
  - Diversidad léxica (n-gramas, Distinct-N, Self-BLEU).
  - Detección IA vs humano.
  - Impacto en tareas de clasificación clínica.
- **Fine-tuning local con LoRA**, aunque no mejoró al modelo base.
- **Consideraciones éticas** y control de riesgos de uso clínico.

## Estructura del repositorio

```
.
│   .env
│   .gitignore
│   README.md
│
├───data
│   ├───diagnoses_df
│   └───generated
├───finetuned_models
│   │   .gitkeep
│   │
│   ├───loras
│   └───merged_models
├───metrics
│       metrics.csv
│
├───notebooks
│       0. llama_finetuning.ipynb
│       1. EDA and Generation.ipynb        
│       2. Authorship_model_selection.ipynb
│       3. Evaluation.ipynb
│       4. Results_Analysis.ipynb
│
├───requirements
│       requirements.in
│       requirements.txt
│
└───tfm
    │   settings.py
    │
    ├───ai_generated_detection
    │       ai_generated_detection_models.py
    │
    ├───llm
    │       base_llm.py
    │       llm_openai.py
    │       llm_vllm.py
    │       __init__.py
    │
    ├───ngrams
    │       ngrams_study.py
    │
    ├───prompt_engineering
    │       data_generation_prompts.py
    │
    ├───train_models
    │       Hugo_Embeddings_and_classification_layer.py
    │       Hugo_Embeddings_to_classification_model.py
    │
    └───utils
            utils.py
```