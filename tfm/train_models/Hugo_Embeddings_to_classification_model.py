import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util
import os, random, ast, json, math, gc, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import warnings, shutil, tempfile
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.utils import shuffle


MODEL_NAME = "intfloat/multilingual-e5-large"
MAX_LEN = 192
BATCH_SIZE = 64
LR = 0.0003465971561151952
EPOCHS_BASELINE = 20
HIDDEN_DIM = 1920

class DiagnosisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt"
        )
        self.input_ids = enc["input_ids"]
        self.attn_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "labels": self.labels[idx],
        }


class BaselineClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=0, num_labels=None):
        super().__init__()
        self.encoder = encoder
        encoder_dim = encoder.config.hidden_size
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_labels)
            )
        else:
            self.classifier = nn.Linear(encoder_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS
        return self.classifier(pooled)
        

def epoch_loop(model, data_loader, criterion, optimizer, device, scheduler=None, train=False):
    model.train() if train else model.eval()
    losses, logits_list, labels_list = [], [], []
    for batch in tqdm(data_loader, desc=f"Batches ({'train' if train else 'eval'})"):
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        labels     = batch["labels"].to(device)

        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, labels)
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if scheduler: scheduler.step()
        losses.append(loss.item())

        logits_list.append(torch.sigmoid(outputs).detach().cpu())
        labels_list.append(labels.detach().cpu())
    y_pred = (torch.vstack(logits_list) > 0.5).int().numpy()
    y_true = torch.vstack(labels_list).numpy()
    return np.mean(losses), f1_score(y_true, y_pred, average="micro")
    
def run_model_training_embeddings_to_classification_model(dataset_name):
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_to_be_evaluated_path = f"../data/generated/{dataset_name}.csv"

    diagnoses_df = pd.read_csv(dataset_to_be_evaluated_path)
    for col in ["Codigos_diagnosticos", "Diagnosticos_estandar"]:
        diagnoses_df[col] = diagnoses_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    mlb = MultiLabelBinarizer()
    data_human = diagnoses_df[diagnoses_df['generated'] == False].reset_index(drop=True)
    data_generated = diagnoses_df[diagnoses_df['generated'] == True].reset_index(drop=True)

    texts_human = ["query: " + t for t in data_human["Descripcion_diagnosticos"].tolist()]
    y_human = mlb.fit_transform(data_human["Diagnosticos_estandar"])

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    X_human = np.array(texts_human)

    for train_idx, tmp_idx in msss.split(np.zeros(len(X_human)), y_human):
        X_train, y_train = X_human[train_idx], y_human[train_idx]
        X_tmp,   y_tmp   = X_human[tmp_idx],  y_human[tmp_idx]

    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=SEED
    )
    for val_idx, test_idx in msss_val.split(np.zeros(len(X_tmp)), y_tmp):
        X_val,  y_val  = X_tmp[val_idx],  y_tmp[val_idx]
        X_test, y_test = X_tmp[test_idx], y_tmp[test_idx]

    X_generated = ["query: " + t for t in data_generated["Descripcion_diagnosticos"].tolist()]
    y_generated = mlb.transform(data_generated["Diagnosticos_estandar"])

    X_train = np.concatenate([X_train, X_generated])
    y_train = np.concatenate([y_train, y_generated])

    X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

    X_train, X_val, X_test = map(lambda a: a.tolist(), [X_train, X_val, X_test])

    train_ratio = y_train.sum(axis=0) / (y_human.sum(axis=0) + y_generated.sum(axis=0))
    val_ratio   = y_val.sum(axis=0)   / (y_human.sum(axis=0) + y_generated.sum(axis=0))
    test_ratio  = y_test.sum(axis=0)  / (y_human.sum(axis=0) + y_generated.sum(axis=0))

    print(f"train: {np.round(train_ratio.mean(), 3)}, val: {np.round(val_ratio.mean(), 3)}, test: {np.round(test_ratio.mean(), 3)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    for p in base_model.parameters(): p.requires_grad = False


    train_ds = DiagnosisDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = DiagnosisDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    test_ds = DiagnosisDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2)

    baseline = BaselineClassifier(base_model, hidden_dim=HIDDEN_DIM, num_labels=y_human.shape[1]).to(device)
    optimizer = torch.optim.AdamW(baseline.classifier.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    for epoch in range(1, EPOCHS_BASELINE+1):
        print(f"🔹 Epoch {epoch:02d} / {EPOCHS_BASELINE}")
        train_loss, train_f1 = epoch_loop(baseline, train_dl, criterion, optimizer, device, train=True)
        val_loss, val_f1     = epoch_loop(baseline, val_dl, criterion, optimizer, device, train=False)
        print(f"Epoch [{epoch:02d}]  train f1={train_f1:.4f} | val f1={val_f1:.4f}")
        best_f1 = max(best_f1, val_f1)

    best_f1_train = epoch_loop(baseline, train_dl, criterion, optimizer, device, train=False)[1]
    best_f1_val = epoch_loop(baseline, val_dl, criterion, optimizer, device, train=False)[1]
    best_f1_test = epoch_loop(baseline, test_dl, criterion, optimizer, device, train=False)[1]
    print(f"🔹 Best F1 (train) baseline: {best_f1_train:.4f}")
    print(f"🔹 Best F1 (val) baseline: {best_f1_val:.4f}")
    print(f"🔹 Best F1 (test) baseline: {best_f1_test:.4f}")

    torch.save({
        "model_state": baseline.state_dict(),
        "tokenizer": MODEL_NAME,
        "mlb_classes": mlb.classes_.tolist()
    }, f"../trained_models/Embeddings_to_classification_model_{dataset_name}.pt")
    print(f"📦 Model saved as optuna/Embeddings_to_classification_model_{dataset_name}.pt")

    del train_ds, val_ds, test_ds
    del train_dl, val_dl, test_dl
    del base_model, baseline
    del optimizer, criterion
    gc.collect()

    return best_f1_train, best_f1_val, best_f1_test