# OpenAI
system_prompt_openai = """
Eres un asistente médico especializado en generar diagnósticos clínicos breves en español.
Solo debes devolver un JSON válido que contenga la clave "diagnosticos" asociada a una lista de cadenas.
No incluyas explicaciones, texto adicional ni comillas triples.
"""


user_prompt_openai = """
A continuación tienes información sobre una etiqueta CIE-10 y ejemplos de diagnósticos asociados:

- Códigos CIE-10 asociados a los ejemplos: {codigos}
- Nombre de la etiqueta: {nombre_codigo}
- Diagnósticos de ejemplo:
{ejemplos}

Tu tarea es generar exactamente {cantidad} diagnósticos clínicos nuevos, distintos de los anteriores,
que correspondan a la etiqueta indicada ({codigos} - {nombre_codigo}).

Devuelve la respuesta como un JSON con esta estructura exacta:

{{
  "diagnosticos": [
    "Diagnóstico A",
    "Diagnóstico B",
    ...
  ]
}}

Debe ser JSON válido, sin ningún texto adicional.
"""

# Llama-3.2-3B-Instruct
system_prompt_llama = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un asistente médico especializado en generar diagnósticos clínicos breves en español.
Devuelve únicamente un JSON **válido y cerrado correctamente** (todas las llaves deben estar balanceadas). No incluyas comillas triples ni texto adicional.
El JSON debe tener esta estructura exacta:

{
  "diagnosticos": ["Diagnóstico A", "Diagnóstico B", ...]
}
"""


user_prompt_llama = """<|eot_id|><|start_header_id|>user<|end_header_id|>
A continuación tienes información sobre una etiqueta CIE-10 y ejemplos de diagnósticos asociados:

- Códigos CIE-10 asociados a los ejemplos: {codigos}
- Nombre de la etiqueta: {nombre_codigo}
- Diagnósticos de ejemplo:
{ejemplos}

Tu tarea es generar exactamente {cantidad} diagnósticos clínicos nuevos, distintos de los anteriores,
que correspondan a la etiqueta indicada ({codigos} - {nombre_codigo}).

Devuelve la respuesta como un JSON con esta estructura exacta:

{{
  "diagnosticos": [
    "Diagnóstico A en una frase breve",
    "Diagnóstico B en una frase breve",
    ...
  ]
}}

Debe ser JSON válido, sin ningún texto adicional.<|eot_id|>
Answer: <|start_header_id|>assistant<|end_header_id|>
"""
