import pandas as pd
import numpy as np

# === Step 1: 加载原始数据 ===
df = pd.read_csv(
    'test.csv',
    sep=',',
    header=0,
    parse_dates=['DateTime'],
    index_col='DateTime',
    low_memory=False
)

# === Step 2: 替换缺失值 "?" 为 NaN 并转换为 float ===
df.replace('?', np.nan, inplace=True)
df = df.astype('float32')

# === Step 3: 用前后24h和48h的值填充缺失 ===
for shift_minutes in [1440, 2880, -1440, -2880]:
    df = df.fillna(df.shift(shift_minutes))

# === Step 4: 添加 sub_metering_4 ===
df['sub_metering_4'] = (df['Global_active_power'] * 1000 / 60) - (
    df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
)

# === Step 5: 保存处理后的分钟级数据 ===
df.to_csv('test_process.csv')

# === Step 6: 按天重采样并聚合 ===
agg_dict = {
    'Global_active_power': 'sum',
    'Global_reactive_power': 'sum',
    'Sub_metering_1': 'sum',
    'Sub_metering_2': 'sum',
    'Voltage': 'mean',
    'Global_intensity': 'mean',
    'RR': 'first',
    'NBJRR1': 'first',
    'NBJRR5': 'first',
    'NBJRR10': 'first',
    'NBJBROU': 'first',
    'sub_metering_4': 'sum'
}

# 用刚才处理好的 df 直接聚合，无需重新读取
daily_data = df.resample('D').agg(agg_dict)

# === Step 7: 保存日聚合数据 ===
daily_data.to_csv('test_days.csv')