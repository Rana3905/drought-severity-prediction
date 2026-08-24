"""
train_model.py -- LSTM + Attention Drought Classifier (v4)
===========================================================
Tuning changes:
  - Input projection layer before LSTM (better feature mixing)
  - Hidden size 96 (balanced capacity)
  - Label smoothing in loss (reduces overconfidence)
  - Warmup + cosine LR schedule
  - Longer patience (30 epochs)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES   = 4
PROJ_SIZE   = 32       # input projection dimension
HIDDEN_SIZE = 96
N_LAYERS    = 1
DROPOUT     = 0.4
BATCH_SIZE  = 256      # large batch for stable gradients with full dataset
LR          = 5e-4
WEIGHT_DECAY= 5e-3
LABEL_SMOOTH= 0.1      # label smoothing factor
MAX_EPOCHS  = 200
PATIENCE    = 30
MAX_CW      = 2.0

print(f"Device: {DEVICE}")

class DroughtDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMAttention(nn.Module):
    def __init__(self, input_size, proj_size, hidden_size, n_layers, dropout, n_classes):
        super().__init__()
        # Input projection: compress 18 features -> proj_size with BN
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size),
            nn.LayerNorm(proj_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm   = nn.LSTM(proj_size, hidden_size, n_layers,
                              batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.drop   = nn.Dropout(dropout)
        self.fc1    = nn.Linear(hidden_size, 32)
        self.fc2    = nn.Linear(32, n_classes)
        self.relu   = nn.ReLU()

    def forward(self, x, return_attn=False):
        # x: (B, T, F)
        B, T, F = x.shape
        x = self.proj(x.reshape(B*T, F)).reshape(B, T, -1)  # (B, T, proj)
        out, _  = self.lstm(x)                               # (B, T, H)
        scores  = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        ctx     = (out * weights.unsqueeze(-1)).sum(dim=1)   # (B, H)
        logits  = self.fc2(self.relu(self.fc1(self.drop(ctx))))
        return (logits, weights) if return_attn else logits

def evaluate(model, loader, criterion):
    model.eval()
    loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss  += criterion(logits, yb).item() * len(xb)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)
    return (loss / len(loader.dataset),
            accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro", zero_division=0),
            mean_absolute_error(trues, preds),
            np.sqrt(mean_squared_error(trues, preds)),
            preds, trues)

def main():
    data    = np.load("sequences.npz", allow_pickle=True)
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]
    X_test  = data["X_test"];  y_test  = data["y_test"]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Train class dist: {np.bincount(y_train, minlength=N_CLASSES)}")
    print(f"Val   class dist: {np.bincount(y_val,   minlength=N_CLASSES)}")
    print(f"Test  class dist: {np.bincount(y_test,  minlength=N_CLASSES)}")

    train_ld = DataLoader(DroughtDataset(X_train, y_train), BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_ld   = DataLoader(DroughtDataset(X_val, y_val), BATCH_SIZE*2, num_workers=0)
    test_ld  = DataLoader(DroughtDataset(X_test, y_test), BATCH_SIZE*2, num_workers=0)

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw), None, MAX_CW).astype(np.float32)
    print(f"Class weights: {np.round(cw, 3)}")

    model     = LSTMAttention(X_train.shape[-1], PROJ_SIZE, HIDDEN_SIZE,
                               N_LAYERS, DROPOUT, N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw).to(DEVICE),
                                     label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (MAX_EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state, no_imp = 0.0, None, 0

    print(f"\n{'Epoch':>6} {'Train':>9} {'Val':>9} {'ValAcc':>7} {'ValF1':>7} {'LR':>8}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tloss = 0.0
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * len(xb)
        tloss /= len(train_ld.dataset)
        scheduler.step()

        vloss, vacc, vf1, *_ = evaluate(model, val_ld, criterion)
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {tloss:>9.4f} {vloss:>9.4f} {vacc:>7.4f} {vf1:>7.4f} {lr_now:>8.2e}")

        # Use val F1 as primary criterion (more meaningful than val loss with class imbalance)
        if vf1 > best_f1:
            best_f1    = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")

    _, acc, f1, mae, rmse, preds, trues = evaluate(model, test_ld, criterion)
    print(f"\n{'='*45}")
    print(f"TEST RESULTS")
    print(f"{'='*45}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(trues, preds, labels=list(range(N_CLASSES))))
    print(f"Classes: 0=None  1=Mild  2=Moderate  3=Severe+")

if __name__ == "__main__":
    main()
