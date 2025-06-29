import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ========== 位置编码 ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return x

# ========== 数据加载与归一化 ==========
def load_datasets(train_file, test_file):
    train_df = pd.read_csv(train_file, parse_dates=['DateTime'], index_col='DateTime')
    test_df = pd.read_csv(test_file, parse_dates=['DateTime'], index_col='DateTime')
    train_df = train_df.fillna(method='ffill')
    test_df = test_df.fillna(method='ffill')

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    return train_scaled, test_scaled, scaler

# ========== 滑动窗口转换 ==========
def to_supervised(data, n_input, n_out=1):
    X, y = [], []
    for i in range(len(data) - n_input - n_out + 1):
        in_end = i + n_input
        out_end = in_end + n_out
        X.append(data[i:in_end, :])
        y.append(data[in_end:out_end, 0])
    return np.array(X), np.array(y)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== Transformer模型 ==========
class TransformerForecaster(nn.Module):
    def __init__(self, n_features, n_outputs, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        x = self.input_proj(x)        # (B, T, d_model)
        x = x.permute(1, 0, 2)       # (T, B, d_model)
        x = self.pos_encoder(x)      # (T, B, d_model)
        encoded = self.encoder(x)    # (T, B, d_model)
        out = encoded[-1]            # (B, d_model) 用最后时间步的编码向量
        return self.decoder(out)     # (B, n_outputs)

# ========== 训练 ==========
def train_model(model, train_loader, epochs=150, lr=1e-4, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return model

# ========== 递归预测评估 ==========
def evaluate_model_recursive(train, test, n_input, n_output, device='cpu'):
    train_x, train_y = to_supervised(train, n_input, n_output)
    train_dataset = SequenceDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = TransformerForecaster(train.shape[1], n_outputs=n_output)
    model = train_model(model, train_loader, device=device)

    history = np.concatenate((train[-n_input:], test), axis=0)
    pred_len = len(test)
    preds_dict = {i: [] for i in range(pred_len)}
    trues = test[:, 0]

    model.eval()
    with torch.no_grad():
        for i in range(pred_len):
            hist_data = history[i:i + n_input].reshape(1, n_input, history.shape[1])
            input_x = torch.tensor(hist_data, dtype=torch.float32).to(device)
            out = model(input_x).cpu().numpy().squeeze()  # (n_output,)
            for j in range(min(n_output, pred_len - i)):
                preds_dict[i + j].append(out[j])
            # 用当前预测值更新history以实现递归预测
            if i + n_input < len(history):
                history[i + n_input, 0] = out[0]  # 用预测第1步值替换下一时刻作为输入

    preds = np.array([np.mean(preds_dict[i]) for i in range(pred_len)])

    return model, preds, trues

# ========== 主函数 ==========
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(42)

    train_data, test_data, scaler = load_datasets('train_days.csv', 'test_days.csv')

    all_preds_90 = []
    mse_list_90, mae_list_90 = [], []

    all_preds_365 = []
    mse_list_365, mae_list_365 = [], []

    for i in range(5):
        print(f"=== Run {i+1}/5: Predict 90 days ===")
        seed_everything(42 + i)
        model, preds, trues = evaluate_model_recursive(train_data, test_data, n_input=90, n_output=90, device=device)

        # 反归一化预测和真实值
        preds_inv = scaler.inverse_transform(np.concatenate([preds.reshape(-1,1), np.zeros((len(preds), train_data.shape[1]-1))], axis=1))[:,0]
        trues_inv = scaler.inverse_transform(np.concatenate([trues.reshape(-1,1), np.zeros((len(trues), train_data.shape[1]-1))], axis=1))[:,0]

        mse = mean_squared_error(trues_inv, preds_inv)
        mae = mean_absolute_error(trues_inv, preds_inv)
        print(f"Run {i+1} (90d): MSE = {mse:.4f}, MAE = {mae:.4f}")
        mse_list_90.append(mse)
        mae_list_90.append(mae)
        all_preds_90.append(preds_inv)

        print(f"=== Run {i+1}/5: Predict 365 days ===")
        seed_everything(84 + i)
        model, preds_365, trues_365 = evaluate_model_recursive(train_data, test_data, n_input=90, n_output=365, device=device)

        preds_365_inv = scaler.inverse_transform(np.concatenate([preds_365.reshape(-1,1), np.zeros((len(preds_365), train_data.shape[1]-1))], axis=1))[:,0]
        trues_365_inv = scaler.inverse_transform(np.concatenate([trues_365.reshape(-1,1), np.zeros((len(trues_365), train_data.shape[1]-1))], axis=1))[:,0]

        mse_365 = mean_squared_error(trues_365_inv, preds_365_inv)
        mae_365 = mean_absolute_error(trues_365_inv, preds_365_inv)
        print(f"Run {i+1} (365d): MSE = {mse_365:.4f}, MAE = {mae_365:.4f}")
        mse_list_365.append(mse_365)
        mae_list_365.append(mae_365)
        all_preds_365.append(preds_365_inv)

    # 保存预测结果和真实值到 CSV
    df_90 = pd.DataFrame({'GroundTruth': trues_inv})
    for i, preds in enumerate(all_preds_90):
        df_90[f'Pred_90_Run_{i+1}'] = preds
    df_90.to_csv("predictions_90days_My_Transformer.csv", index=False)

    df_365 = pd.DataFrame({'GroundTruth': trues_365_inv})
    for i, preds in enumerate(all_preds_365):
        df_365[f'Pred_365_Run_{i+1}'] = preds
    df_365.to_csv("predictions_365days_My_Transformer.csv", index=False)

    # 绘图
    plt.figure(figsize=(14, 6))
    plt.plot(df_90['GroundTruth'], label='GT 90d', linewidth=2, color='black')
    for i in range(5):
        plt.plot(df_90[f'Pred_90_Run_{i+1}'], label=f'90d Run {i+1}', alpha=0.6)
    plt.title("Predictions over 90 Days", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_90days_My_Transformer.png", dpi=300)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(df_365['GroundTruth'], label='GT 365d', linewidth=2, color='black')
    for i in range(5):
        plt.plot(df_365[f'Pred_365_Run_{i+1}'], label=f'365d Run {i+1}', alpha=0.6)
    plt.title("Predictions over 365 Days", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_365days_My_Transformer.png", dpi=300)
    plt.close()

    # 打印统计
    print("\n=== Evaluation Summary ===")
    print(f"[90 days] MSE: mean = {np.mean(mse_list_90):.4f}, std = {np.std(mse_list_90):.4f}")
    print(f"[90 days] MAE: mean = {np.mean(mae_list_90):.4f}, std = {np.std(mae_list_90):.4f}")
    print(f"[365 days] MSE: mean = {np.mean(mse_list_365):.4f}, std = {np.std(mse_list_365):.4f}")
    print(f"[365 days] MAE: mean = {np.mean(mae_list_365):.4f}, std = {np.std(mae_list_365):.4f}")