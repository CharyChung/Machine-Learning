import pandas as pd
import matplotlib.pyplot as plt

# 读取预测结果CSV
df_90 = pd.read_csv("/public/home/zhouxiabing/data/yzhang/ML/predictions_90days_Transformer.csv")
df_365 = pd.read_csv("/public/home/zhouxiabing/data/yzhang/ML/predictions_365days_Transformer.csv")

# 绘制90天预测对比图
plt.figure(figsize=(14, 6))
plt.plot(df_90['GroundTruth'], label='Ground Truth 90d', color='dimgray', linewidth=1.2)
for i in range(1, 6):
    plt.plot(df_90[f'Pred_90_Run_{i}'], label=f'90d Run {i}', alpha=0.6)
plt.title("Predictions over 90 Days", fontsize=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_90days_Transformer_updated.png", dpi=300)
plt.show()

# 绘制365天预测对比图
plt.figure(figsize=(14, 6))
plt.plot(df_365['GroundTruth'], label='Ground Truth 365d', color='dimgray', linewidth=1.2)
for i in range(1, 6):
    plt.plot(df_365[f'Pred_365_Run_{i}'], label=f'365d Run {i}', alpha=0.6)
plt.title("Predictions over 365 Days", fontsize=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_365days_Transformer_updated.png", dpi=300)
plt.show()