import pandas as pd
import matplotlib.pyplot as plt

# 读取三个csv文件
df1 = pd.read_csv('/public/home/zhouxiabing/data/yzhang/ML/predictions_365days_LSTM.csv')
df2 = pd.read_csv('/public/home/zhouxiabing/data/yzhang/ML/predictions_365days_Transformer.csv')
df3 = pd.read_csv('/public/home/zhouxiabing/data/yzhang/ML/predictions_365days_My_Transformer.csv')

# 取三个文件第一列（GroundTruth）平均，生成新的 GroundTruth 列
groundtruth_mean = (df1.iloc[:, 0] + df2.iloc[:, 0] + df3.iloc[:, 0]) / 3

# 定义函数计算每个文件中预测列的平均值
def mean_preds(df):
    pred_cols = [col for col in df.columns if col.startswith('Pred_365_Run')]
    return df[pred_cols].mean(axis=1)

# 计算每个文件预测均值列
df1['Pred_Mean'] = mean_preds(df1)
df2['Pred_Mean'] = mean_preds(df2)
df3['Pred_Mean'] = mean_preds(df3)

# 画图
plt.figure(figsize=(14, 6))
plt.plot(groundtruth_mean, label='GroundTruth', color='black', linewidth=1.2)

plt.plot(df1['Pred_Mean'], label='LSTM', alpha=0.7)
plt.plot(df2['Pred_Mean'], label='Transformer', alpha=0.7)
plt.plot(df3['Pred_Mean'], label='Transformer&Position', alpha=0.7)

plt.title("Average GroundTruth and Prediction Means from 3 Files")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("average_groundtruth_and_predictions_365days.png", dpi=300)