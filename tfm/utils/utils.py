import itertools
import math
import random
import re
import json
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generar_grupos_para_llamadas(
    ejemplos_disponibles: list[str],
    faltantes: int,
    max_ejemplos_input: int = 5,
    generar_por_llamada: int = 5
) -> list[list[str]]:
    """
    Construye, para una etiqueta dada, las listas de ejemplos que se enviarán
    al LLM en cada llamada, de modo que:
      - En cada llamada se envíen hasta `max_ejemplos_input` diagnósticos de ejemplo.
      - Cada llamada devuelve `generar_por_llamada` nuevos diagnósticos (hasta cubrir
        el total de `faltantes`).
      - Se aprovechan al máximo los diagnósticos disponibles sin repetirlos más veces
        de lo necesario, y cuando haya pocos disponibles (< max_ejemplos_input), se
        combinan en subconjuntos de tamaño decreciente para maximizar la diversidad.

    Parámetros:
    -----------
    ejemplos_disponibles : list[str]
        Lista de diagnósticos (textos) ya existentes para esta etiqueta.
    faltantes : int
        Cuántos diagnósticos sintéticos faltan por generar en total.
    max_ejemplos_input : int (por defecto 5)
        Número máximo de ejemplos de entrada que se incluirán en cada prompt.
    generar_por_llamada : int (por defecto 5)
        Número de diagnósticos que se esperan obtener de cada llamada al LLM.

    Retorna:
    --------
    list[list[str]]
        Una lista de longitud `N_calls`, donde cada elemento es otra lista con
        entre 1 y `max_ejemplos_input` strings. Cada sublista son los diagnósticos
        de ejemplo que luego se pasan al LLM en una llamada separada.
    """

    disponibles = len(ejemplos_disponibles)
    if faltantes <= 0:
        return []  # No hay que generar nada

    # 1) Calcular cuántas llamadas al LLM hacen falta en total
    N_calls = math.ceil(faltantes / generar_por_llamada)

    # 2) Caso: suficientes ejemplos disponibles para llenar max_ejemplos_input en cada llamada
    if disponibles >= max_ejemplos_input:
        # ¿Cuántos "slots" de ejemplo necesitamos en total?
        slots_necesarios = N_calls * max_ejemplos_input

        # Construir un pool repetido y barajado hasta alcanzar 'slots_necesarios'
        pool = []
        while len(pool) < slots_necesarios:
            random.shuffle(ejemplos_disponibles)
            pool.extend(ejemplos_disponibles)

        # Cortar exactamente a 'slots_necesarios'
        pool = pool[:slots_necesarios]

        # Partir el pool en chunks de tamaño 'max_ejemplos_input'
        grupos = []
        for i in range(N_calls):
            inicio = i * max_ejemplos_input
            fin = inicio + max_ejemplos_input
            subgrupo = pool[inicio:fin]  # max_ejemplos_input distintos
            grupos.append(subgrupo)

        return grupos

    # 3) Caso: disponibles < max_ejemplos_input
    #    Generar todas las combinaciones de tamaño k = disponibles, disponibles-1, ..., 1
    #    pero solo hasta k <= max_ejemplos_input (aunque aquí disponibles < max_ejemplos_input)
    todos_combos = []
    for k in range(disponibles, 0, -1):  # disponibles, disponibles-1, ..., 1
        if k > max_ejemplos_input:
            continue
        for combo in itertools.combinations(ejemplos_disponibles, k):
            todos_combos.append(list(combo))

    # Si hay suficientes combinaciones para cubrir todas las llamadas sin repetir:
    if len(todos_combos) >= N_calls:
        return todos_combos[:N_calls]

    # Si no alcanzan, primero usamos cada combo una vez:
    grupos = []
    for combo in todos_combos:
        grupos.append(combo)
        if len(grupos) >= N_calls:
            return grupos

    # Nos quedan aún N_calls - len(grupos) llamadas por cubrir
    faltan = N_calls - len(grupos)

    # Barajamos los combos y los reutilizamos en orden hasta completar N_calls
    mezcla = todos_combos.copy()
    random.shuffle(mezcla)
    idx = 0
    while faltan > 0:
        grupos.append(mezcla[idx % len(mezcla)])
        idx += 1
        faltan -= 1

    return grupos


def asignacion_proporcional(group_sizes: list[int], total_faltantes: int) -> list[int]:
    """Devuelve una lista de enteros, la cantidad a generar por cada grupo,
    usando el método de restos mayores para que la suma sea total_faltantes."""
    S = sum(group_sizes)
    if S == 0:
        # Si no hay ejemplos en ningún grupo, repartir uniformemente
        base = total_faltantes // len(group_sizes)
        resto = total_faltantes % len(group_sizes)
        return [base + (1 if i < resto else 0) for i in range(len(group_sizes))]

    # Cálculo de cuotas reales y partes enteras
    cuotas = [size / S * total_faltantes for size in group_sizes]
    enteros = [math.floor(q) for q in cuotas]
    restantes = total_faltantes - sum(enteros)

    # Partes fraccionarias y orden para asignar los ‘restantes’
    fracs = sorted(
        [(cuotas[i] - enteros[i], i) for i in range(len(cuotas))],
        reverse=True
    )
    for k in range(restantes):
        _, idx = fracs[k]
        enteros[idx] += 1

    return enteros


def extraer_json_valido(texto_modelo):
    try:
        match = re.search(r'\{.*\}', texto_modelo, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON malformado: {e}")
        else:
            # 1) Buscamos la posición de la primera llave de apertura
            inicio = texto_modelo.find('{')
            if inicio == -1:
                raise ValueError("No se encontró un JSON válido.")
            
            # 2) Recorremos desde esa posición para encontrar el cierre balanceado
            contador = 0
            fin = None
            for i, ch in enumerate(texto_modelo[inicio:], start=inicio):
                if ch == '{':
                    contador += 1
                elif ch == '}':
                    contador -= 1
                # Cuando contador vuelve a 0, hemos cerrado todas las llaves
                if contador == 0:
                    fin = i
                    break
            
            # 3) Extraemos el fragmento JSON
            if fin is not None:
                # Tenemos un cierre completo
                json_str = texto_modelo[inicio:fin+1]
            else:
                # Nunca cerró todas, tomamos todo lo que queda y añadimos cierres
                json_str = texto_modelo[inicio:]
                # contador > 0 indica cuántas '{' quedaron sin emparejar
                json_str += '}' * contador
            
            # 4) Intentamos parsear
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON malformado tras corrección: {e}")
            
    except Exception as e:
        raise ValueError(f"Error al extraer JSON: {e}")


def procesar_grupo(grupo, etiquetas, nombres_etiquetas, cantidad_a_generar, system_prompt, user_prompt, llm, max_retries, temperature, frequency_penalty, presence_penalty):
    print(f"\t\t - {len(grupo)} -> {grupo}")
    retries = 0
    while retries < max_retries:
        llm_raw_response = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt.format(
                codigos=etiquetas,
                nombre_codigo=nombres_etiquetas,
                ejemplos=grupo,
                cantidad=cantidad_a_generar
            ),
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        try:
            diagnosticos_generados_json = extraer_json_valido(llm_raw_response)
            diagnosticos_generados = diagnosticos_generados_json.get("diagnosticos", [])
            print(f"\t\t   Diagnósticos generados parcial: {len(diagnosticos_generados)}")
            return diagnosticos_generados
        except Exception as e:
            retries += 1
            print(f"\t\t   Error al parsear JSON ({e}); intento {retries}/{max_retries}")

    raise RuntimeError(f"No se pudo procesar el grupo {grupo} tras {max_retries} intentos")


def clean_text(text):
    # Convert to string and lowercase
    text = str(text).lower()

    # Remove accents and diacritics
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

    # Replace newlines, tabs, and multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep letters, numbers, spaces, and periods
    text = re.sub(r'[^a-zñ0-9. ]', '', text)

    # Trim leading and trailing whitespace
    text = text.strip()

    return text


def clean_cie10_code(code):
    """Normalize a CIE10 code to a standardized format.

    - Adds '.0' to codes that start with 'F' and contain only digits afterward (e.g., 'F3' becomes 'F3.0').
    - Removes a trailing '0' after the decimal in codes like 'F3.10' to make them 'F3.1'.
    - Leaves codes unchanged if they don't match either pattern.

    Parameters:
        code (str): A CIE10 code as a string.

    Returns:
        str: The normalized CIE10 code.
    """
    # If key starts with 'F' and the rest are digits and no dot, change key to have '.0'
    if code.startswith("F") and code[1:].isdigit() and "." not in code:
        return f"{code}.0"
    # If key matches pattern F<digits>.<digits>0 (e.g., F3.10), convert to F<digits>.<digits> (e.g., F3.1)
    elif (
        code.startswith("F")
        and "." in code
        and code.endswith("0")
        and not code.endswith(".0")
    ):
        return code[:-1]
    else:
        return code


def read_cie10_file(file_path):
    """Read a CSV file containing CIE10 variable-label mappings and returns a complete mapping dictionary.

    The function processes the CSV to exclude entries with label "<none>", constructs a dictionary
    mapping CIE10 codes to their descriptions, adds additional predefined mappings, and normalizes
    CIE10 codes by appending '.0' to those that start with 'F' and are followed by digits without a period.

    Parameters:
        file_path (str): The path to the CSV file containing CIE10 variables and their labels.
                         Expected columns are "Variable" and "Label".

    Returns:
        dict: A dictionary mapping CIE10 codes (as strings) to their corresponding descriptions.
    """
    vars_df = pd.read_csv(file_path)

    vars_df = vars_df[vars_df["Label"] != "<none>"]
    cie10_map = dict(zip(vars_df["Variable"], vars_df["Label"]))

    # Add the CIE10 values that are not in the vars_df
    CIE10_values_not_in_vars = {
        "F32": "Episodio depresivo",
        "F33": "Trastorno depresivo mayor, recurrente",
        "F4": "Trastorno de ansiedad, disociativo, relacionado con estrés y otros trastornos mentales somatomorfos no psicóticos",
        "F40": "Trastornos de ansiedad fóbica",
        "F50": "Trastornos de la conducta alimentaria",
        "COGNITIV": "Dimensión cognitiva",
        "FAM_APO": "Problemas y conflictos familiares",
        "LAB_MOB": "Problemas y conflictos laborales",
        "No_DX": "No diagnóstico",
        "PAREJ": "Problemas y conflictos de pareja",
        "altas_capacidades": "Altas capacidades intelectuales",
    }
    cie10_map.update(CIE10_values_not_in_vars)

    # Clean the cie10 codes
    cie10_map = {clean_cie10_code(key): value for key, value in cie10_map.items()}

    return cie10_map

def plot_corr_columns(data, grupo_1, grupo_2):
    corr = pd.DataFrame(index=grupo_1, columns=grupo_2)

    for col1 in grupo_1:
        for col2 in grupo_2:
            corr.loc[col1, col2] = data[col1].corr(data[col2])

    plt.figure(figsize=(15, 6))
    ax = sns.heatmap(corr.astype(float), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()
