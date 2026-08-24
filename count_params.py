import torch.nn as nn

class LSTMAttention(nn.Module):
    def __init__(self, input_size, soil_size, proj_size, hidden_size,
                 n_layers, dropout, soil_proj, n_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size), nn.LayerNorm(proj_size),
            nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.lstm = nn.LSTM(proj_size, hidden_size, n_layers,
                             batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
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
        pass  # not needed just to count params

def count_and_print(model, label):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- {label} ---")
    print(f"Total parameters: {total}")
    print(f"Total parameters (Millions): {total / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable}")
    print(f"Trainable parameters (Millions): {trainable / 1e6:.2f} M")

# US model config (from train_model.py): input_size=18, soil_size=31
us_model = LSTMAttention(input_size=18, soil_size=31, proj_size=32,
                          hidden_size=96, n_layers=1, dropout=0.4,
                          soil_proj=16, n_classes=4)
count_and_print(us_model, "US LSTM+Attention")

# Kazakhstan model config (from train_model_kazakhstan.py): input_size=1, soil_size=31
kz_model = LSTMAttention(input_size=1, soil_size=31, proj_size=16,
                          hidden_size=48, n_layers=1, dropout=0.4,
                          soil_proj=8, n_classes=4)
count_and_print(kz_model, "Kazakhstan LSTM+Attention")