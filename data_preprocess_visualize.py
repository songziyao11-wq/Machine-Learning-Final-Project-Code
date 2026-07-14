import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 1. 基础路径与配置 --------------------------
root_path = r"C:\Users\23768\Desktop\机器学习数据集"
# 读取上一步拼接好的总数据集
merged_df = pd.read_csv(os.path.join(root_path, "merged_wind_dataset.csv"))
# 解决中文显示乱码问题（Windows系统适配）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 2. 时序数据预处理（划分前准备） --------------------------
# 时间戳转为标准时间格式，按时间升序排序（时序数据核心要求）
merged_df["Timestamp"] = pd.to_datetime(merged_df["Timestamp"])
df_sorted = merged_df.sort_values("Timestamp").reset_index(drop=True)
# 提取所有原始数值特征列（排除时间戳，共15个）
numeric_cols = [col for col in df_sorted.columns if col != "Timestamp"]

# -------------------------- 3. 严格7:2:1时序数据集划分 --------------------------
# 时序数据禁止随机打乱，按时间先后划分：前70%训练，中间20%验证，最后10%测试
total_samples = len(df_sorted)
train_cut = int(total_samples * 0.7)
val_cut = int(total_samples * 0.9)

train_set = df_sorted.iloc[:train_cut].copy()
val_set = df_sorted.iloc[train_cut:val_cut].copy()
test_set = df_sorted.iloc[val_cut:].copy()

# 打印划分结果
print("="*60)
print("数据集划分完成（严格按时间顺序7:2:1）：")
print(f"训练集：{len(train_set)} 条  占比 {len(train_set)/total_samples*100:.1f}%")
print(f"验证集：{len(val_set)} 条  占比 {len(val_set)/total_samples*100:.1f}%")
print(f"测试集：{len(test_set)} 条  占比 {len(test_set)/total_samples*100:.1f}%")
print("="*60)

# -------------------------- 4. 数据清洗（无数据泄露规范） --------------------------
# 4.1 缺失值处理
# 训练集：线性插值（利用时序前后关联性，仅用训练集自身数据）
train_set = train_set.interpolate(method='linear', limit_direction='both')
# 验证/测试集：前向填充为主（模拟真实场景，不能使用未来数据），后向填充兜底
val_set = val_set.ffill().bfill()
test_set = test_set.ffill().bfill()

# 打印缺失值处理结果
print("\n缺失值处理完成，处理后各数据集缺失值总数：")
print(f"训练集：{train_set.isnull().sum().sum()} 个")
print(f"验证集：{val_set.isnull().sum().sum()} 个")
print(f"测试集：{test_set.isnull().sum().sum()} 个")

# 4.2 异常值处理（3σ原则，仅用训练集统计量，避免数据泄露）
# 基于训练集计算均值和标准差，确定3σ上下边界
train_stats = train_set[numeric_cols].agg(['mean', 'std']).T
train_stats['upper_bound'] = train_stats['mean'] + 3 * train_stats['std']
train_stats['lower_bound'] = train_stats['mean'] - 3 * train_stats['std']

# 对三个数据集执行盖帽处理（异常值替换为边界值，保留样本时序连续性）
def cap_outliers(df, stats_df, cols):
    df_copy = df.copy()
    for col in cols:
        upper = stats_df.loc[col, 'upper_bound']
        lower = stats_df.loc[col, 'lower_bound']
        df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
    return df_copy

train_clean = cap_outliers(train_set, train_stats, numeric_cols)
val_clean = cap_outliers(val_set, train_stats, numeric_cols)
test_clean = cap_outliers(test_set, train_stats, numeric_cols)

print("\n异常值处理完成（3σ盖帽法），边界仅基于训练集计算")
print("="*60)

# 4.3 构造日周期时间特征（仅2个，总特征数15+2=17）
def add_time_cycle_features(df):
    df_copy = df.copy()
    # 仅提取小时，构造日周期正弦-余弦编码
    df_copy['hour'] = df_copy['Timestamp'].dt.hour
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
    # 删除辅助列
    df_copy.drop(['hour'], axis=1, inplace=True)
    return df_copy

# 给三个数据集同步添加时间特征
train_clean = add_time_cycle_features(train_clean)
val_clean = add_time_cycle_features(val_clean)
test_clean = add_time_cycle_features(test_clean)

# 更新数值特征列表，加入2个日周期特征，总维度17
time_feature_list = ['hour_sin', 'hour_cos']
numeric_cols = numeric_cols + time_feature_list
print(f"日周期特征构造完成，新增2个特征，当前总输入特征数：{len(numeric_cols)}（17维）")
print("="*60)

# 4.4 保存清洗后的数据集（原文件名、路径完全不变）
train_clean.to_csv(os.path.join(root_path, "train.csv"), index=False, encoding="utf-8-sig")
val_clean.to_csv(os.path.join(root_path, "val.csv"), index=False, encoding="utf-8-sig")
test_clean.to_csv(os.path.join(root_path, "test.csv"), index=False, encoding="utf-8-sig")
print("清洗后的训练/验证/测试集已保存到数据集文件夹")

# -------------------------- 5. 数据集可视化（所有图名、路径完全不变） --------------------------
# 5.1 不同高度风速时序分布图
plt.figure(figsize=(14, 6))
plt.plot(train_clean['Timestamp'], train_clean['WindSpeed_10m'], label='10米高度风速', alpha=0.8, linewidth=0.8)
plt.plot(train_clean['Timestamp'], train_clean['WindSpeed_50m'], label='50米高度风速', alpha=0.8, linewidth=0.8)
plt.plot(train_clean['Timestamp'], train_clean['WindSpeed_100m'], label='100米高度风速', alpha=0.8, linewidth=0.8)
plt.title('不同高度风速时序分布（训练集）', fontsize=14)
plt.xlabel('时间', fontsize=12)
plt.ylabel('风速 (m/s)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(root_path, "风速时序分布图.png"), dpi=300, bbox_inches='tight')
plt.close()
# ========== 计算训练集三个高度风速的描述性统计（含偏度、峰度） ==========
height_list = ['10m', '50m', '100m']
col_list = ['WindSpeed_10m', 'WindSpeed_50m', 'WindSpeed_100m']

stats_df = pd.DataFrame()
for i in range(3):
    col = col_list[i]
    # 计算统计量：kurt()+3 是将超额峰度转为皮尔逊峰度（正态分布为3）
    stats_df[height_list[i]] = [
        round(train_clean[col].mean(), 2),
        round(train_clean[col].std(), 2),
        round(train_clean[col].max(), 2),
        round(train_clean[col].skew(), 2),
        round(train_clean[col].kurt() + 3, 2)
    ]
stats_df.index = ['均值', '标准差', '最大值', '偏度', '峰度']
print("\n训练集不同高度风速描述性统计：")
print(stats_df)
# ==========================================================================
# 5.2 风速数值分布直方图（带核密度曲线）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
height_list = ['10m', '50m', '100m']
for i, height in enumerate(height_list):
    col = f'WindSpeed_{height}'
    sns.histplot(train_clean[col], kde=True, ax=axes[i], bins=30, color='#1f77b4', alpha=0.7)
    axes[i].set_title(f'{height}高度风速分布', fontsize=13)
    axes[i].set_xlabel('风速 (m/s)', fontsize=11)
    axes[i].set_ylabel('样本数量', fontsize=11)
    axes[i].grid(alpha=0.3)
plt.suptitle('风速数值概率分布图（训练集）', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(root_path, "风速分布直方图.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5.3 特征相关性热力图（自动包含2个时间特征，图名、路径完全不变）
plt.figure(figsize=(12, 10))
corr_matrix = train_clean[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, square=True)
plt.title('特征相关性热力图（训练集）', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(root_path, "特征相关性热力图.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n可视化图表已全部生成并保存到数据集文件夹，共3张：")
print("1. 风速时序分布图.png")
print("2. 风速分布直方图.png")
print("3. 特征相关性热力图.png")
print("="*60)