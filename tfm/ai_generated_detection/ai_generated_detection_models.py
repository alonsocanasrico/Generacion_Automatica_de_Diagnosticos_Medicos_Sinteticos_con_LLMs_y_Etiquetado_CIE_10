import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from transformers import pipeline
import pandas as pd
import re
from html import unescape

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    LongformerTokenizer,
    LongformerForSequenceClassification,
    PreTrainedModel,
)

from sklearn.metrics import accuracy_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Wangkevin02AIDetectorModel:
    def __init__(self, model_name="wangkevin02/AI_Detect_Model", max_length=4096):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.tokenizer.padding_side = "right"

    @torch.no_grad()
    def get_probability(self, texts, batch_size=8):
        all_probs = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(batch_texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length,
                                    return_tensors='pt').to(self.device)

            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1)
            all_probs.append(probabilities.cpu())  # move back to CPU to avoid GPU memory overflow

        return torch.cat(all_probs, dim=0)
    

class DesklibAIDetectorModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_emb = torch.sum(last_hidden_state * mask_exp, dim=1)
        sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        pooled = sum_emb / sum_mask

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())

        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}

    @torch.no_grad()
    def predict(
        self,
        texts,
        tokenizer: AutoTokenizer,
        device: torch.device,
        batch_size: int = 8,
        max_len: int = 768,
        threshold: float = 0.5
    ):
        """
        Predice sobre un texto o una lista de textos.
        - Si `texts` es str: devuelve (prob, etiqueta).
        - Si `texts` es lista: devuelve ([probs], [etiquetas]).
        """
        # Normalizar a lista
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        self.to(device).eval()
        all_probs, all_labels = [], []

        for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = self(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].squeeze(-1)
            probs = torch.sigmoid(logits)
            labels = (probs >= threshold).long()

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        if single:
            return all_probs[0], all_labels[0]
        return all_probs, all_labels


class PrasoonmhwrAIDetectorModel:
    def __init__(self, model_name_or_path="prasoonmhwr/ai_detection_model", device=None, max_length=128):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, texts, batch_size=8, threshold=0.5):
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        all_probs = []
        all_labels = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length,
                                    return_tensors="pt").to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(-1)
            labels = (probs >= threshold).long()

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        if single_input:
            return all_probs[0], all_labels[0]
        return all_probs, all_labels

class FakespotAIDetectorModel:
    def __init__(self, column_name='Descripcion_diagnosticos', batch_size=32):
        self.column_name = column_name
        self.batch_size = batch_size
        self.classifier = pipeline(
            "text-classification",
            model="fakespot-ai/roberta-base-ai-text-detection-v1"
        )

    def clean_markdown(self, md_text):
        # Remove code blocks
        md_text = re.sub(r'```.*?```', '', md_text, flags=re.DOTALL)
        # Remove inline code
        md_text = re.sub(r'`[^`]*`', '', md_text)
        # Remove images
        md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
        # Remove links but keep link text
        md_text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', md_text)
        # Remove bold and italic (groups of *, _)
        md_text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md_text)
        md_text = re.sub(r'(\*|_)(.*?)\1', r'\2', md_text)
        # Remove headings
        md_text = re.sub(r'#+ ', '', md_text)
        # Remove blockquotes
        md_text = re.sub(r'^>.*$', '', md_text, flags=re.MULTILINE)
        # Remove list markers
        md_text = re.sub(r'^(\s*[-*+]|\d+\.)\s+', '', md_text, flags=re.MULTILINE)
        # Remove horizontal rules
        md_text = re.sub(r'^\s*[-*_]{3,}\s*$', '', md_text, flags=re.MULTILINE)
        # Remove tables
        md_text = re.sub(r'\|.*?\|', '', md_text)
        # Remove raw HTML tags
        md_text = re.sub(r'<.*?>', '', md_text)
        # Decode HTML entities
        md_text = unescape(md_text)
        return md_text

    def clean_text(self, t):
        t = self.clean_markdown(t)
        t = t.replace("\n"," ")
        t = t.replace("\t"," ")
        t = t.replace("^M"," ")
        t = t.replace("\r"," ")
        t = t.replace(" ,", ",")
        t = re.sub(" +", " ", t)
        return t

    def _batch(self, iterable, batch_size):
        """Divide una lista en batches."""
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]

    def predict(self, df):
        if self.column_name not in df.columns:
            raise ValueError(f"La columna '{self.column_name}' no existe en el DataFrame")

        texts = df[self.column_name].astype(str).apply(self.clean_text).tolist()
        predictions = []

        for batch in tqdm(self._batch(texts, self.batch_size), total=len(texts)//self.batch_size + 1, desc="Clasificando"):
            results = self.classifier(batch, truncation=True)
            # Mapear etiqueta a 1 si es texto generado por IA, 0 si no
            binary = [1 if r['label'].lower() == 'ai' else 0 for r in results]
            predictions.extend(binary)

        return predictions



def calculate_ai_human_metrics(preds_indices, data_df):
    preds = preds_indices.numpy()  # pasar a numpy
    labels = data_df['generated'].values.astype(int)
    accuracy = accuracy_score(labels, preds)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy