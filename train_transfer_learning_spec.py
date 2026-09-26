import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

RESULTS_PATH = "results.json"
US_WEIGHTS_PATH = "model_us_full_archmatched.pt"
DATA_PATH = "sequences_kazakhstan.npz"
CLASS_LABELS = ["None", "Mild", "Moderate", "Severe+"]
N_CLASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PROJ_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ = 24, 64, 1, 0.4, 8
BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTH = 64, 5e-4, 5e-3, 0.1
FT_LR, FT_WEIGHT_DECAY = 5e-5, 5e-3
MAX_EPOCHS, PATIENCE, MAX_CW = 120, 30, 2.0

def save_results(model_name, metrics, extra=None):
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    entry = {k: float(v) for k, v in metrics.items()}
    entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
    entry["n_classes"] = N_CLASSES
    entry["seed"] = SEED
    if extra:
        entry.update(extra)
    existing = all_results.get(model_name, [])
    if isinstance(existing, dict):
        existing = [existing]
    existing.append(entry)
    all_results[model_name] = existing
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved run #{len(existing)} for '{model_name}' -> {RESULTS_PATH}")


class DroughtDataset(Dataset):
    def __init__(self, X, S, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.S[i], self.y[i]


class LSTMAttention(nn.Module):
    def __init__(self, input_size, soil_size, proj_size, hidden_size,
                 n_layers, dropout, soil_proj, n_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size), nn.LayerNorm(proj_size),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.lstm = nn.LSTM(proj_size, hidden_size, n_layers, batch_first=True)
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.soil_proj = nn.Sequential(
            nn.Linear(soil_size, soil_proj), nn.LayerNorm(soil_proj),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.fc1 = nn.Linear(hidden_size + soil_proj, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, s):
        B, T, F_ = x.shape
        x = self.proj(x.reshape(B*T, F_)).reshape(B, T, -1)
        out, _ = self.lstm(x)
        scores = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)
        soil_ctx = self.soil_proj(s)
        combined = torch.cat([self.drop(ctx), soil_ctx], dim=1)
        return self.fc2(self.relu(self.fc1(combined)))


def reinit_input_projection(model):
    """Re-initialize proj.0 (Linear input_size->PROJ_SIZE) with the default
    nn.Linear init, as it has a different input width than the US model."""
    lin = model.proj[0]
    nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
    if lin.bias is not None:
        fan_in = lin.weight.size(1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(lin.bias, -bound, bound)


def load_us18_weights(model, path):
    """Load US 18-feature weights into the KZ model except the input
    projection (24x18 vs 24x1). Return number of params loaded."""
    state = torch.load(path, map_location="cpu")
    transferred = {k: v for k, v in state.items()
                   if k not in ("proj.0.weight", "proj.0.bias")}
    model.load_state_dict(transferred, strict=False)
    reinit_input_projection(model)
    return sum(v.numel() for v in transferred.values())


def freeze_except(model, keep_prefixes=("proj.0",)):
    n_trainable = 0
    for name, p in model.named_parameters():
        keep = any(name.startswith(k) for k in keep_prefixes)
        p.requires_grad_(keep)
        if keep:
            n_trainable += p.numel()
    return n_trainable


def evaluate(model, loader, criterion):
    model.eval()
    loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for xb, sb, yb in loader:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb, sb)
            loss += criterion(logits, yb).item() * len(xb)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)
    return (loss / len(loader.dataset),
            accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro", zero_division=0),
            mean_absolute_error(trues, preds),
            np.sqrt(mean_squared_error(trues, preds)),
            preds, trues)


def train(model, train_ld, val_ld, criterion, trainable_params, lr, wd,
          model_name, extra, save_path=None):
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (MAX_EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state, no_imp = 0.0, None, 0
    print(f"\nTRAINING: {model_name} (lr={lr}, trainable={extra.get('trainable_params')})")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tloss = 0.0
        for xb, sb, yb in train_ld:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb, sb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            tloss += loss.item() * len(xb)
        tloss /= len(train_ld.dataset)
        scheduler.step()

        vloss, vacc, vf1, *_ = evaluate(model, val_ld, criterion)
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {tloss:>9.4f} {vloss:>9.4f} {vacc:>7.4f} {vf1:>7.4f}")

        if vf1 > best_f1:
            best_f1, best_state, no_imp = vf1, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved {save_path}")

    _, acc, f1, mae, rmse, preds, trues = evaluate(model, test_ld, criterion)
    print(f"\n{'='*60}\nTEST RESULTS: {model_name}\n{'='*60}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    cm = confusion_matrix(trues, preds, labels=list(range(N_CLASSES)))
    print(f"Confusion Matrix:\n{cm}")

    extra = dict(extra)
    extra["confusion_matrix"] = cm.tolist()
    extra["n_features"] = int(X_train.shape[-1])
    extra["real_data"] = True
    save_results(model_name, {"Accuracy": acc, "Macro F1": f1,
                              "MAE": mae, "RMSE": rmse}, extra=extra)
    return f1


LOCK_PATH = "train_spec.lock"

def acquire_lock():
    stale = 12 * 3600
    if os.path.exists(LOCK_PATH):
        age = datetime.now().timestamp() - os.path.getmtime(LOCK_PATH)
        if age < stale:
            print(f"Lock file {LOCK_PATH} exists (age {age:.0f}s) -- "
                  "another instance is running. Exiting.")
            raise SystemExit(0)
        print(f"Removing stale lock ({age:.0f}s old).")
        os.remove(LOCK_PATH)
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def main():
    acquire_lock()
    try:
        _run()
    finally:
        release_lock()


def _run():
    global X_train, test_ld
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found.")
    if not os.path.exists(US_WEIGHTS_PATH):
        raise SystemExit(f"{US_WEIGHTS_PATH} not found -- train the arch-matched "
                         "US 18-feature model first.")
    data = np.load(DATA_PATH, allow_pickle=True)
    X_train, y_train, S_train = data["X_train"], data["y_train"], data["S_train"]
    X_val, y_val, S_val = data["X_val"], data["y_val"], data["S_val"]
    X_test, y_test, S_test = data["X_test"], data["y_test"], data["S_test"]

    print(f"Device: {DEVICE} | SEED: {SEED}")
    print(f"Data: {X_train.shape} train | {X_val.shape} val | {X_test.shape} test")

    train_ld = DataLoader(DroughtDataset(X_train, S_train, y_train), BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(DroughtDataset(X_val, S_val, y_val), BATCH_SIZE*2)
    test_ld = DataLoader(DroughtDataset(X_test, S_test, y_test), BATCH_SIZE*2)

    cw_vals = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = np.clip(np.sqrt(cw_vals), None, MAX_CW).astype(np.float32)
    cw_t = torch.tensor(cw).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=LABEL_SMOOTH)

    # ---- A. From scratch (KZ 1-feat, arch 24/64) ---------------------- #
    print("\n" + "#"*60)
    print("# A. FROM SCRATCH -- KZ 1-feature, PROJ=24/HIDDEN=64")
    print("#"*60)
    mA = LSTMAttention(X_train.shape[-1], S_train.shape[-1], PROJ_SIZE,
                       HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                       N_CLASSES).to(DEVICE)
    n_train_a = sum(p.numel() for p in mA.parameters())
    f1_a = train(mA, train_ld, val_ld, criterion,
                 list(mA.parameters()), LR, WEIGHT_DECAY,
                 "LSTM+Attention (KZ, 1-feat, from scratch 24/64)",
                 {"transfer_learning": False, "trainable_params": n_train_a},
                 save_path="model_kazakhstan_1feat_scratch_2464.pt")

    # ---- B. SPEC-COMPLIANT: US18 pretrained, ONLY input layer tuned ---- #
    print("\n" + "#"*60)
    print("# B. US18 TRANSFER, FROZEN -- spec-compliant")
    print("#    only proj.0 (layer seeing the single SPEI feature) trains")
    print("#"*60)
    mB = LSTMAttention(X_train.shape[-1], S_train.shape[-1], PROJ_SIZE,
                       HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                       N_CLASSES).to(DEVICE)
    n_loaded = load_us18_weights(mB, US_WEIGHTS_PATH)
    n_train_b = freeze_except(mB, keep_prefixes=("proj.0",))
    trainable_b = [p for p in mB.parameters() if p.requires_grad]
    f1_b = train(mB, train_ld, val_ld, criterion, trainable_b, LR, WEIGHT_DECAY,
                 "LSTM+Attention (KZ, 1-feat, US18 transfer frozen)",
                 {"transfer_learning": True, "pretrained_source": "US 18-feat (arch-matched)",
                  "transferred_params": n_loaded, "trainable_params": n_train_b,
                  "frozen": "all except proj.0", "spec_compliant": True},
                 save_path="model_kazakhstan_1feat_us18_frozen.pt")

    # ---- C. US18 pretrained, full fine-tune (non-spec contrast) -------- #
    print("\n" + "#"*60)
    print("# C. US18 TRANSFER, FULL FINE-TUNE -- non-spec contrast")
    print("#    everything unfrozen at low LR (mirrors earlier 14-feat attempt)")
    print("#"*60)
    mC = LSTMAttention(X_train.shape[-1], S_train.shape[-1], PROJ_SIZE,
                       HIDDEN_SIZE, N_LAYERS, DROPOUT, SOIL_PROJ,
                       N_CLASSES).to(DEVICE)
    n_loaded_c = load_us18_weights(mC, US_WEIGHTS_PATH)
    n_train_c = sum(p.numel() for p in mC.parameters())
    f1_c = train(mC, train_ld, val_ld, criterion,
                 list(mC.parameters()), FT_LR, FT_WEIGHT_DECAY,
                 "LSTM+Attention (KZ, 1-feat, US18 transfer full-finetune)",
                 {"transfer_learning": True, "pretrained_source": "US 18-feat (arch-matched)",
                  "transferred_params": n_loaded_c, "trainable_params": n_train_c,
                  "frozen": "none", "spec_compliant": False, "ft_lr": FT_LR},
                 save_path="model_kazakhstan_1feat_us18_fullft.pt")

    print("\n" + "="*60)
    print("SUMMARY (Macro F1):")
    print(f"  A. From scratch              : {f1_a:.4f}")
    print(f"  B. US18 frozen (spec)        : {f1_b:.4f}")
    print(f"  C. US18 full fine-tune       : {f1_c:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()