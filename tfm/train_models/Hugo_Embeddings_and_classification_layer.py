import ast
import gc
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from contextlib import nullcontext


warnings.filterwarnings("ignore", category=UserWarning)


# === Custom classifier model ===
class DiagnosisClassifier(nn.Module):
    def __init__(self, base_model, base_model_output_dim=1024, hidden_dim=768, num_labels=10):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(base_model_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(pooled_output)

# === Dataset wrapper ===
class DiagnosisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def epoch_loop(model, data_loader, criterion, optimizer, device, scheduler=None, train=False): # TODO mover
        model.train() if train else model.eval()
        losses, logits_list, labels_list = [], [], []

        grad_ctx = nullcontext() if train else torch.no_grad() # NEW
        with grad_ctx:
            for batch in tqdm(data_loader, desc=f"Batches ({'train' if train else 'eval'})"):
                input_ids  = batch["input_ids"].to(device)
                attn_mask  = batch["attention_mask"].to(device)
                labels     = batch["labels"].to(device)

                outputs = model(input_ids, attn_mask)
                loss = criterion(outputs, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # NEW
                    optimizer.step()
                    if scheduler: scheduler.step()

                losses.append(loss.item())
                logits_list.append(torch.sigmoid(outputs).detach().cpu())
                labels_list.append(labels.detach().cpu())

        y_pred = (torch.vstack(logits_list) > 0.5).int().numpy()
        y_true = torch.vstack(labels_list).numpy()
        return np.mean(losses), f1_score(y_true, y_pred, average="micro")



def run_model_training_embeddings_and_classification_layer(dataset_name):
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_to_be_evaluated_path = f"../data/generated/{dataset_name}.csv"

    diagnoses_df = pd.read_csv(dataset_to_be_evaluated_path)
    for col in ["Codigos_diagnosticos", "Diagnosticos_estandar"]:
        diagnoses_df[col] = diagnoses_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    diagnoses_df = diagnoses_df.head(100) # TODO: quitar esta línea - For testing purposes, limit to 100 rows
    diagnoses_df

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(diagnoses_df["Diagnosticos_estandar"])
    # Add the prefix required by e5-Large
    texts_prefixed = ["query: " + t for t in diagnoses_df["Descripcion_diagnosticos"].tolist()]

    # Random splits on multilabel data, ensuring that each label's distribution is preserved across training and test sets.
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    X = np.array(texts_prefixed)

    for train_idx, tmp_idx in msss.split(np.zeros(len(X)), y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_tmp,   y_tmp   = X[tmp_idx],   y[tmp_idx]

    # 50-50 over the 30 % left ⇒ 15 %/15 %
    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=SEED
    )
    for val_idx, test_idx in msss_val.split(np.zeros(len(X_tmp)), y_tmp):
        X_val,  y_val  = X_tmp[val_idx],  y_tmp[val_idx]
        X_test, y_test = X_tmp[test_idx], y_tmp[test_idx]

    # Convert to lists for compatibility with SentenceTransformer
    X_train, X_val, X_test = map(lambda a: a.tolist(), [X_train, X_val, X_test])

    # Convert to numpy arrays and float32 for PyTorch compatibility
    y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32)

    # Check that each label maintains its ratio approx.
    train_ratio = y_train.sum(axis=0) / y.sum(axis=0)
    val_ratio   = y_val.sum(axis=0)   / y.sum(axis=0)
    test_ratio  = y_test.sum(axis=0)  / y.sum(axis=0)

    print(f"train: {np.round(train_ratio.mean(), 3)}, val: {np.round(val_ratio.mean(), 3)}, test: {np.round(test_ratio.mean(), 3)}")

    # === Load model and tokenizer ===
    MODEL_NAME = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    EPOCHS_BASELINE = 50
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-2
    BATCH_SIZE = 16
    HIDDEN_DIM = 1024
    MAX_LEN = 256

    train_ds = DiagnosisDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = DiagnosisDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    test_ds = DiagnosisDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # === Initialize model ===
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    for p in base_model.parameters(): p.requires_grad = True
    baseline = DiagnosisClassifier(base_model=base_model, hidden_dim=HIDDEN_DIM, num_labels=y.shape[1]).to(device)

    # === Training setup ===
    # optimizer = torch.optim.AdamW(baseline.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # NEW
    optimizer = torch.optim.AdamW([
        {"params": baseline.base.parameters(), "lr": 2e-5},         # encoder
        {"params": baseline.classifier.parameters(), "lr": 1e-3}    # new head
    ], weight_decay=WEIGHT_DECAY)
    from transformers import get_linear_schedule_with_warmup
    num_training_steps = len(train_dl) * EPOCHS_BASELINE
    num_warmup_steps  = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    # criterion = nn.BCEWithLogitsLoss()
    pos_counts = y_train.sum(axis=0) # NEW
    neg_counts = y_train.shape[0] - pos_counts # NEW
    pos_weight = torch.tensor((neg_counts + 1) / (pos_counts + 1), dtype=torch.float32).to(device) # NEW
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # NEW

    best_f1 = 0
    for epoch in range(1, EPOCHS_BASELINE+1):
        print(f"🔹 Epoch {epoch:02d} / {EPOCHS_BASELINE}")
        train_loss, train_f1 = epoch_loop(baseline, train_dl, criterion, optimizer, device, scheduler=scheduler, train=True)
        val_loss, val_f1     = epoch_loop(baseline, val_dl, criterion, optimizer, device, scheduler=scheduler, train=False)
        print(f"Epoch [{epoch:02d}]  train loss={train_loss:.4f} | val loss={val_loss:.4f}")
        print(f"Epoch [{epoch:02d}]  train f1={train_f1:.4f} | val f1={val_f1:.4f}")
        best_f1 = max(best_f1, val_f1)

    best_f1_train = epoch_loop(baseline, train_dl, criterion, optimizer, device, train=False)[1]
    best_f1_val = epoch_loop(baseline, val_dl, criterion, optimizer, device, train=False)[1]
    best_f1_test = epoch_loop(baseline, test_dl, criterion, optimizer, device, train=False)[1]
    print(f"🔹 Best F1 (train) baseline: {best_f1_train:.4f}")
    print(f"🔹 Best F1 (val) baseline: {best_f1_val:.4f}")
    print(f"🔹 Best F1 (test) baseline: {best_f1_test:.4f}")

    # Save the model
    torch.save({
        "model_state": baseline.state_dict(),
        "tokenizer": MODEL_NAME,
        "mlb_classes": mlb.classes_.tolist()
    }, f"../trained_models/Embeddings_and_classification_layer_{dataset_name}.pt")
    print(f"📦 Model saved as optuna/Embeddings_and_classification_layer_{dataset_name}.pt")

    del train_ds, val_ds, test_ds
    del train_dl, val_dl, test_dl
    del base_model, baseline
    del optimizer, pos_counts, neg_counts, pos_weight, criterion
    gc.collect()

    return best_f1_train, best_f1_val, best_f1_test