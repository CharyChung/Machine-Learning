import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random

# ========== 数据加载 ==========
def load_datasets(train_file, test_file):
    train_df = pd.read_csv(train_file, parse_dates=['DateTime'], index_col='DateTime')
    test_df = pd.read_csv(test_file, parse_dates=['DateTime'], index_col='DateTime')
    train_df = train_df.fillna(method='ffill')
    test_df = test_df.fillna(method='ffill')

    train = train_df.values
    test = test_df.values

    return train, test
# ========== 滑动窗口转换 ==========
def to_supervised(data, n_input, n_out=1):
    X, y = [], []
    for i in range(len(data) - n_input - n_out + 1):
        in_end = i + n_input
        out_end = in_end + n_out
        if out_end < len(data):
            X.append(data[i:in_end, :])
            y.append(data[in_end:out_end, 0])
    return np.array(X), np.array(y)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== 模型定义 ==========
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, n_features, n_outputs, hidden_dim=256):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.repeat = nn.Linear(hidden_dim, n_outputs * hidden_dim)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        _, (hidden, _) = self.encoder(x)
        repeated = self.repeat(hidden[-1])
        repeated = repeated.view(batch_size, -1, hidden.size(-1))
        decoded_output, _ = self.decoder(repeated)
        out = self.dense(decoded_output)
        return out

# ========== 模型训练 ==========
def train_model(model, train_loader, epochs=500, lr=1e-4, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(epochs)):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
# ========== 评估 ==========
def evaluate_model_recursive(train, test, n_input, n_output, device='cpu'):
    train_x, train_y = to_supervised(train, n_input, n_out=n_output)
    train_dataset = SequenceDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # Step 2: 模型训练
    model = EncoderDecoderLSTM(train.shape[-1], n_outputs=n_output).to(device)
    model = train_model(model, train_loader, device=device)

    # Step 3: 滚动预测整个 test
    history = np.concatenate((train[-n_input:], test), axis=0)
    pred_len = len(test)
    preds = {i: [] for i in range(pred_len)}
    trues = test[:, 0]  # 真实值是 test 的第一列

    with torch.no_grad():
        for i in range(len(test)):
            hist_data = history[i:i + n_input].reshape(1, n_input, history.shape[1])
            input_x = torch.tensor(hist_data, dtype=torch.float32).to(device)
            out = model(input_x).cpu().numpy().squeeze(-1)  # shape (7,)
            for j in range(min(n_output, pred_len - i)):
                preds[i + j].append(out[0, j])
    preds = np.array([np.mean(preds[i]) for i in range(pred_len)])
    trues = np.array(trues)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    return mse, mae, preds, trues

# ========== 主程序 ==========
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # seed_everything(42)
    train_data, test_data = load_datasets('train_days.csv', 'test_days.csv')

    all_preds_90 = []
    mse_list_90, mae_list_90 = [], []

    all_preds_365 = []
    mse_list_365, mae_list_365 = [], []

    for i in range(5):
        print(f"=== Run {i+1}/5: Predict 90 days ===")
        # seed_everything(42 + i)
        mse, mae, preds, trues = evaluate_model_recursive(train_data, test_data, n_input=90, n_output=90, device=device)
        print(f"Run {i+1} (90d): MSE = {mse:.4f}, MAE = {mae:.4f}")
        mse_list_90.append(mse)
        mae_list_90.append(mae)
        all_preds_90.append(preds)

    for i in range(5):
        print(f"=== Run {i+1}/5: Predict 365 days ===")
        # seed_everything(84 + i)  # 不同种子
        mse, mae, preds, trues_365 = evaluate_model_recursive(train_data, test_data, n_input=90, n_output=365, device=device)
        print(f"Run {i+1} (365d): MSE = {mse:.4f}, MAE = {mae:.4f}")
        mse_list_365.append(mse)
        mae_list_365.append(mae)
        all_preds_365.append(preds)

    # 保存预测结果和真实值到 CSV
    df_90 = pd.DataFrame({'GroundTruth': trues})
    for i, preds in enumerate(all_preds_90):
        df_90[f'Pred_90_Run_{i+1}'] = preds
    df_90.to_csv("predictions_90days_LSTM.csv", index=False)

    df_365 = pd.DataFrame({'GroundTruth': trues_365})
    for i, preds in enumerate(all_preds_365):
        df_365[f'Pred_365_Run_{i+1}'] = preds
    df_365.to_csv("predictions_365days_LSTM.csv", index=False)

    # 绘图
    plt.figure(figsize=(14, 6))
    plt.plot(df_90['GroundTruth'], label='GT 90d', linewidth=2, color='black')
    for i in range(5):
        plt.plot(df_90[f'Pred_90_Run_{i+1}'], label=f'90d Run {i+1}', alpha=0.6)
    plt.title("Predictions over 90 Days", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_90days_LSTM.png", dpi=300)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(df_365['GroundTruth'], label='GT 365d', linewidth=2, color='black')
    for i in range(5):
        plt.plot(df_365[f'Pred_365_Run_{i+1}'], label=f'365d Run {i+1}', alpha=0.6)
    plt.title("Predictions over 365 Days", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_365days_LSTM.png", dpi=300)
    plt.close()

    # 打印统计
    print("\n=== Evaluation Summary ===")
    print(f"[90 days] MSE: mean = {np.mean(mse_list_90):.4f}, std = {np.std(mse_list_90):.4f}")
    print(f"[90 days] MAE: mean = {np.mean(mae_list_90):.4f}, std = {np.std(mae_list_90):.4f}")
    print(f"[365 days] MSE: mean = {np.mean(mse_list_365):.4f}, std = {np.std(mse_list_365):.4f}")
    print(f"[365 days] MAE: mean = {np.mean(mae_list_365):.4f}, std = {np.std(mae_list_365):.4f}")