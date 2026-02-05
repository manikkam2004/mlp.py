# ============================================================
# Advanced Time Series Forecasting with Neural State Space Models (SSMs)
# Refined Implementation with Efficiency Benchmarking
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# -------------------------------
# 1. DATA GENERATION / LOADING
# -------------------------------
# Synthetic dataset rationale: simulate high-frequency sensor/financial signal
np.random.seed(42)
time_idx = np.arange(0, 5000)
signal = np.sin(0.02 * time_idx) + 0.5*np.sin(0.05*time_idx) + np.random.normal(0,0.1,len(time_idx))
df = pd.DataFrame({"time": time_idx, "value": signal})
df["value"] = (df["value"] - df["value"].mean()) / df["value"].std()

# -------------------------------
# 2. BASELINE MODEL (XGBoost)
# -------------------------------
def create_lag_features(series, lags=10):
    df_feat = pd.DataFrame({"y": series})
    for lag in range(1, lags+1):
        df_feat[f"lag_{lag}"] = df_feat["y"].shift(lag)
    return df_feat.dropna()

lags = 10
features = create_lag_features(df["value"], lags)
X = features.drop("y", axis=1).values
y = features["y"].values

tscv = TimeSeriesSplit(n_splits=5)
baseline_preds, baseline_true = [], []

start_time = time.time()
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    baseline_preds.extend(preds)
    baseline_true.extend(y_test)
baseline_train_time = time.time() - start_time

baseline_mae = mean_absolute_error(baseline_true, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(baseline_true, baseline_preds))

# -------------------------------
# 3. NEURAL STATE SPACE MODEL (Closer to S4)
# -------------------------------
class SimpleS4(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, seq_len=50):
        super(SimpleS4, self).__init__()
        # State-space convolution kernel approximation
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim)*0.01)
        self.C = nn.Parameter(torch.randn(input_dim, hidden_dim)*0.01)
        self.seq_len = seq_len
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.A.size(0))
        for t in range(self.seq_len):
            h = h + torch.matmul(h, self.A) + torch.matmul(x[:,t,:], self.B.T)
        out = torch.matmul(h, self.C.T)
        return self.fc(out)

# Prepare sequences
seq_len = 50
series = df["value"].values
data, labels = [], []
for i in range(len(series)-seq_len):
    data.append(series[i:i+seq_len])
    labels.append(series[i+seq_len])
data = np.array(data); labels = np.array(labels)

train_size = int(0.8*len(data))
X_train, X_test = data[:train_size], data[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# Train SSM
model_ssm = SimpleS4()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_ssm.parameters(), lr=0.001)

epochs = 5
start_time = time.time()
for epoch in range(epochs):
    model_ssm.train()
    optimizer.zero_grad()
    output = model_ssm(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
ssm_train_time = time.time() - start_time

# Evaluate
model_ssm.eval()
start_time = time.time()
with torch.no_grad():
    preds_ssm = model_ssm(X_test_t).squeeze().numpy()
    true_ssm = y_test_t.squeeze().numpy()
ssm_infer_time = time.time() - start_time

ssm_mae = mean_absolute_error(true_ssm, preds_ssm)
ssm_rmse = np.sqrt(mean_squared_error(true_ssm, preds_ssm))

# -------------------------------
# 4. RESULTS COMPARISON
# -------------------------------
print("\n=== Performance Comparison ===")
print(f"Baseline (XGBoost) - MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, Train Time: {baseline_train_time:.2f}s")
print(f"SSM (Simplified S4) - MAE: {ssm_mae:.4f}, RMSE: {ssm_rmse:.4f}, Train Time: {ssm_train_time:.2f}s, Inference Time: {ssm_infer_time:.2f}s")
