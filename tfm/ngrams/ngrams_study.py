from collections import Counter
import re
import numpy as np
import sacrebleu

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # quitar puntuación
    return re.sub(r'\s+', ' ', text).strip()

def ngrams_from_text(text, n):
    tokens = text.split()
    return zip(*[tokens[i:] for i in range(n)])

# Función para evaluar métricas globales de n-gramas
def evaluar_ngramas(human_texts, ai_texts, n):
    ngramas_human = Counter()
    ngramas_ai = Counter()

    for txt in human_texts:
        txt_norm = normalize_text(txt)
        ngramas_human.update(ngrams_from_text(txt_norm, n))

    for txt in ai_texts:
        txt_norm = normalize_text(txt)
        ngramas_ai.update(ngrams_from_text(txt_norm, n))

    total_ai = sum(ngramas_ai.values())
    total_human = sum(ngramas_human.values())
    comunes = sum(c for ng, c in ngramas_ai.items() if ng in ngramas_human)

    unicos_ai = len(ngramas_ai)
    unicos_human = len(ngramas_human)
    unicos_ai_prop = unicos_ai / len(ai_texts)
    unicos_human_prop = unicos_human / len(human_texts)
    novedad = unicos_ai - sum(1 for ng in ngramas_ai if ng in ngramas_human)

    frac_comunes_ai = comunes / total_ai if total_ai > 0 else 0.0
    frac_comunes_human = comunes / total_human if total_human > 0 else 0.0

    bleu_lbl = bleu_corpus_weighted(human_texts, ai_texts)
    one_minus_bleu = 1 - bleu_lbl
    novelty_lbl = novelty_n(human_texts, ai_texts, n)
    distinct_n_lbl = distinct_n(ai_texts, n)
    selfbleu_lbl = self_bleu(ai_texts)

    return {
        f'frac_comunes_ai_{n}gram': frac_comunes_ai, # Fracción de n-gramas sintéticos que aparecen en reales. Mide qué tan “parecidos” son los textos sintéticos a los reales. Valores bajos indican que hay mucha novedad en el texto generado, pero podría también ser ruido.
        f'frac_comunes_human_{n}gram': frac_comunes_human, # Fracción de n-gramas reales cubiertos por sintéticos. Indica qué tanto los textos sintéticos cubren la variedad que hay en los textos reales. Si es muy bajo, la generación puede estar “perdiendo” mucha diversidad real.
        # Cuántos n-gramas únicos tiene cada conjunto. En proporción a la cantidad de diagnósticos (si no se hace proporción, muy probablemente tendrá más el conjunto que más diagnósticos tenga). Un número alto indica riqueza léxica y estructural. Comparar estos números te da una idea del “tamaño del vocabulario” en forma de frases cortas.
        f'unicos_ai_prop_{n}gram': unicos_ai_prop,
        f'unicos_human_prop_{n}gram': unicos_human_prop,
        f'novedad_{n}gram': novedad, # te indica cuántos n-gramas se generan que no están en los reales. Un buen balance: ni cero novedad (copiar demasiado), ni demasiada novedad (que puede significar ruido o incoherencias).
        f'one_minus_bleu_{n}gram': one_minus_bleu,
        f'novelty_n_{n}gram': novelty_lbl,
        f'distinct_n_{n}gram': distinct_n_lbl,
        f'self_bleu_{n}gram': selfbleu_lbl
    }

def bleu_corpus_weighted(human_texts, ai_texts):
    """BLEU corpus con pesos custom (uni+bi por defecto)."""
    refs = [human_texts]
    bleu = sacrebleu.corpus_bleu(
        ai_texts, refs,
        force=True, lowercase=True,
        smooth_method="exp"
    )
    return bleu.score / 100.0  # normalizado 0–1

def novelty_n(human_texts, ai_texts, n):
    """
    Proporción de n-gramas en AI que NO aparecen en humanos.
    (complemento directo de frac_comunes_ai).
    """
    human_ngrams = set()
    for txt in human_texts:
        toks = normalize_text(txt).split()
        human_ngrams |= set(ngrams_from_text(" ".join(toks), n))

    total = novel = 0
    for txt in ai_texts:
        toks = normalize_text(txt).split()
        ngs = list(ngrams_from_text(" ".join(toks), n))
        total += len(ngs)
        novel += sum(1 for gram in ngs if gram not in human_ngrams)
    return novel/total if total>0 else 0.0

def distinct_n(texts, n):
    """
    Distinct-N: diversidad absoluta dentro de AI.
    """
    total = 0
    unique = set()
    for txt in texts:
        toks = normalize_text(txt).split()
        ngs = list(ngrams_from_text(" ".join(toks), n))
        total += len(ngs)
        unique.update(ngs)
    return len(unique)/total if total>0 else 0.0

def self_bleu(ai_texts):
    """
    Self-BLEU: media de BLEU de cada AI vs. el resto de AIs.
    """
    scores = []
    for i, hypo in enumerate(ai_texts):
        refs = [t for j,t in enumerate(ai_texts) if j != i]
        bleu = sacrebleu.corpus_bleu(
            [hypo], [refs],
            force=True, lowercase=True,
            smooth_method="exp"
        )
        scores.append(bleu.score / 100.0)
    return float(np.mean(scores))