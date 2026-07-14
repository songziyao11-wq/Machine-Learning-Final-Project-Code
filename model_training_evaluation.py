# ===================== 1. 导入依赖包 =====================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ===================== 2. 全局配置与超参数 =====================
# 固定随机种子，保证实验可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(42)

# 路径配置（完全沿用原路径，无改动）
root_path = r"C:\Users\23768\Desktop\机器学习数据集"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 时序窗口参数（严格符合作业要求：8小时历史窗口）
look_back = 48  # 8小时 = 8*6=48个时间步（10分钟/步）
# 三类预测任务的预测步长（完全符合作业定义）
pred_steps = {
    "单步预测": 1,          # 下一时刻（10分钟后）
    "多步预测A": 6,         # 未来1小时 = 6个时间步
    "多步预测B": 96         # 未来16小时 = 16*6=96个时间步
}

# 目标变量：预测100米高度风速
target_col = "WindSpeed_100m"

# ========== 统一17维特征：15维原始气象+风速 + 2维日周期特征 ==========
all_cols = pd.read_csv(os.path.join(root_path, "train.csv")).columns.tolist()
# 2个日周期时间特征
time_features = ['hour_sin', 'hour_cos']
# 15维原始基础特征
base_features = [col for col in all_cols if col != "Timestamp" and col not in time_features]
# 全部任务统一使用17维全特征
full_features = base_features + time_features
feature_cols = full_features
print(f"全局统一特征维度：{len(feature_cols)}（15维原始特征 + 2维日周期特征）")
# ============================================================

# 深度学习训练超参数（和你原始版本完全一致，无改动）
batch_size = 32
epochs = 30
learning_rate = 0.0005

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 3. 核心工具函数 =====================
# 3.1 时序滑窗数据集构造
def create_sliding_window(data, target_idx, look_back, pred_steps):
    X, y = [], []
    for i in range(look_back, len(data) - pred_steps + 1):
        X.append(data[i-look_back:i, :])
        y.append(data[i:i+pred_steps, target_idx])
    return np.array(X), np.array(y)

# 3.2 统一计算四个评估指标
def calc_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": round(mse, 4), "RMSE": round(float(rmse), 4), "MAE": round(mae, 4), "R²": round(r2, 4)}

# ===================== 4. 数据加载与标准化 =====================
# 加载预处理好的数据集
train_df = pd.read_csv(os.path.join(root_path, "train.csv"))
val_df = pd.read_csv(os.path.join(root_path, "val.csv"))
test_df = pd.read_csv(os.path.join(root_path, "test.csv"))

# 提取17维特征数据
train_data = train_df[feature_cols].values
val_data = val_df[feature_cols].values
test_data = test_df[feature_cols].values

# 特征标准化（仅用训练集拟合，无数据泄露）
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# 获取目标变量的列索引
target_idx = feature_cols.index(target_col)

# ===================== 5. 传统机器学习模型（线性回归+随机森林） =====================
def train_traditional_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx, task_name):
    print(f"--- 开始训练【{model_name}】模型 ---")
    # 三维时序特征展平为二维
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 初始化模型
    if model_name == "线性回归":
        model = LinearRegression()
    elif model_name == "随机森林":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError("不支持的传统模型")

    # 训练模型
    model.fit(X_train_flat, y_train)

    # 预测 + 统一二维格式
    y_pred_scaled = model.predict(X_test_flat)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    # 反标准化还原原始风速尺度
    def inverse_transform_y(y_scaled, scaler, target_idx, n_features):
        y_flat = y_scaled.flatten()
        temp = np.zeros((len(y_flat), n_features))
        temp[:, target_idx] = y_flat
        return scaler.inverse_transform(temp)[:, target_idx].reshape(y_scaled.shape)

    n_features = scaler.n_features_in_
    y_test_true = inverse_transform_y(y_test, scaler, target_idx, n_features)
    y_test_pred = inverse_transform_y(y_pred_scaled, scaler, target_idx, n_features)

    # 计算评估指标
    metrics = calc_metrics(y_test_true.flatten(), y_test_pred.flatten())

    # 保存模型为.pth格式
    save_name = f"{model_name.replace(' ', '_')}_{task_name}.pth"
    torch.save(model, os.path.join(root_path, save_name))
    print(f"--- 【{model_name}】模型训练完成，已保存参数文件 ---")

    # 返回预测均值适配可视化
    return model, metrics, y_test_true.mean(axis=1), y_test_pred.mean(axis=1)

# ===================== 6. 深度学习模型（结构、参数和原始版本完全一致） =====================
# 6.1 构建PyTorch数据加载器
def build_dataloader(X, y, batch_size, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 6.2 LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, pred_steps=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, pred_steps)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        last_step_out = self.dropout(last_step_out)
        return self.fc(last_step_out)

# 6.3 Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=1, pred_steps=1, seq_len=48, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, pred_steps)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        trans_out = self.transformer_encoder(x)
        trans_out = trans_out.permute(0, 2, 1)
        pooled_out = self.pool(trans_out).squeeze(-1)
        return self.fc(pooled_out)

# 6.4 深度学习模型训练与评估函数
def train_deep_model(model_name, train_loader, val_loader, input_size, pred_steps, epochs, lr, device, task_name):
    print(f"\n--- 开始训练【{model_name}】模型，总轮次：{epochs} ---")
    if model_name == "LSTM":
        model = LSTMModel(input_size=input_size, pred_steps=pred_steps).to(device)
    elif model_name == "Transformer":
        model = TransformerModel(input_size=input_size, pred_steps=pred_steps, seq_len=look_back).to(device)
    else:
        raise ValueError("不支持的深度学习模型")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_model_state = None

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证集评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")

        # 保存最优模型（早停机制）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    # 加载最优模型参数
    model.load_state_dict(best_model_state)
    save_name = f"{model_name}_{task_name}.pth"
    torch.save(model.state_dict(), os.path.join(root_path, save_name))
    print(f"--- 【{model_name}】模型训练完成，最优验证损失: {best_val_loss:.4f}，已保存参数文件 ---")

    return model

# 6.5 深度学习模型测试集评估
def evaluate_deep_model(model, test_loader, scaler, target_idx, device):
    model.eval()
    y_pred_all = []
    y_true_all = []
    n_features = scaler.n_features_in_

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu().numpy()
            y_pred_all.append(pred)
            y_true_all.append(batch_y.numpy())

    y_pred_scaled = np.concatenate(y_pred_all, axis=0)
    y_true_scaled = np.concatenate(y_true_all, axis=0)

    def inverse_transform_full(y_scaled):
        batch_size, step_num = y_scaled.shape
        y_flat = y_scaled.flatten()
        temp = np.zeros((len(y_flat), n_features))
        temp[:, target_idx] = y_flat
        y_true_flat = scaler.inverse_transform(temp)[:, target_idx]
        return y_true_flat.reshape(batch_size, step_num)

    y_true_full = inverse_transform_full(y_true_scaled)
    y_pred_full = inverse_transform_full(y_pred_scaled)
    metrics = calc_metrics(y_true_full.flatten(), y_pred_full.flatten())

    return metrics, y_true_full.mean(axis=1), y_pred_full.mean(axis=1)

# ===================== 7. 全任务批量运行（全部统一17维特征） =====================
all_results = {}
all_predictions = {}

for task_name, step in pred_steps.items():
    print(f"\n{'='*70}\n开始执行【{task_name}】任务（预测步长：{step}）\n{'='*70}")

    # 构造滑窗数据集（全部使用统一的17维特征）
    X_train, y_train = create_sliding_window(train_scaled, target_idx, look_back, step)
    X_val, y_val = create_sliding_window(val_scaled, target_idx, look_back, step)
    X_test, y_test = create_sliding_window(test_scaled, target_idx, look_back, step)
    print(f"数据集构造完成：训练集{X_train.shape[0]}条，验证集{X_val.shape[0]}条，测试集{X_test.shape[0]}条")

    # ---------- 传统模型训练 ----------
    lr_model, lr_metrics, lr_true, lr_pred = train_traditional_model(
        "线性回归", X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx, task_name
    )
    rf_model, rf_metrics, rf_true, rf_pred = train_traditional_model(
        "随机森林", X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx, task_name
    )
    print(f"\n线性回归 {task_name} 测试集指标：{lr_metrics}")
    print(f"随机森林 {task_name} 测试集指标：{rf_metrics}")

    # ---------- 深度学习模型训练 ----------
    train_loader = build_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size, shuffle=False)
    test_loader = build_dataloader(X_test, y_test, batch_size, shuffle=False)
    input_size = X_train.shape[2]

    lstm_model = train_deep_model("LSTM", train_loader, val_loader, input_size, step, epochs, learning_rate, device, task_name)
    lstm_metrics, lstm_true, lstm_pred = evaluate_deep_model(lstm_model, test_loader, scaler, target_idx, device)

    trans_model = train_deep_model("Transformer", train_loader, val_loader, input_size, step, epochs, learning_rate, device, task_name)
    trans_metrics, trans_true, trans_pred = evaluate_deep_model(trans_model, test_loader, scaler, target_idx, device)

    print(f"\nLSTM {task_name} 测试集指标：{lstm_metrics}")
    print(f"Transformer {task_name} 测试集指标：{trans_metrics}")

    # 汇总结果
    all_results[task_name] = {
        "线性回归": lr_metrics,
        "随机森林": rf_metrics,
        "LSTM": lstm_metrics,
        "Transformer": trans_metrics
    }
    all_predictions[task_name] = {
        "真实值": lr_true,
        "线性回归": lr_pred,
        "随机森林": rf_pred,
        "LSTM": lstm_pred,
        "Transformer": trans_pred
    }

# 保存所有指标结果为CSV
metrics_df = pd.DataFrame()
for task, models in all_results.items():
    temp = pd.DataFrame(models).T
    temp.insert(0, "预测任务", task)
    metrics_df = pd.concat([metrics_df, temp])
metrics_df.to_csv(os.path.join(root_path, "模型评估指标汇总.csv"), encoding="utf-8-sig")
print(f"\n{'='*70}")
print("所有模型评估指标已保存为：模型评估指标汇总.csv")

# ===================== 8. 可视化生成（所有图名、路径完全不变） =====================
# 8.1 单步预测对比图
plt.figure(figsize=(16, 7))
plot_num = 200
x_axis = range(plot_num)
plt.plot(x_axis, all_predictions["单步预测"]["真实值"][:plot_num], label="真实值", color="black", linewidth=1.5)
plt.plot(x_axis, all_predictions["单步预测"]["线性回归"][:plot_num], label="线性回归", alpha=0.8)
plt.plot(x_axis, all_predictions["单步预测"]["随机森林"][:plot_num], label="随机森林", alpha=0.8)
plt.plot(x_axis, all_predictions["单步预测"]["LSTM"][:plot_num], label="LSTM", alpha=0.8)
plt.plot(x_axis, all_predictions["单步预测"]["Transformer"][:plot_num], label="Transformer", alpha=0.8)
plt.title("单步风速预测：真实值与多模型预测值对比（测试集）", fontsize=14, pad=15)
plt.xlabel("测试集样本序号", fontsize=12)
plt.ylabel("风速 (m/s)", fontsize=12)
plt.legend(ncol=5, loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(root_path, "单步预测结果对比图.png"), dpi=300, bbox_inches="tight")
plt.close()

# 8.2 多步预测A对比图
plt.figure(figsize=(16, 7))
plt.plot(x_axis, all_predictions["多步预测A"]["真实值"][:plot_num], label="真实值", color="black", linewidth=1.5)
plt.plot(x_axis, all_predictions["多步预测A"]["线性回归"][:plot_num], label="线性回归", alpha=0.8)
plt.plot(x_axis, all_predictions["多步预测A"]["随机森林"][:plot_num], label="随机森林", alpha=0.8)
plt.plot(x_axis, all_predictions["多步预测A"]["LSTM"][:plot_num], label="LSTM", alpha=0.8)
plt.plot(x_axis, all_predictions["多步预测A"]["Transformer"][:plot_num], label="Transformer", alpha=0.8)
plt.title("多步预测A（未来1小时）：真实值与多模型预测值对比（测试集）", fontsize=14, pad=15)
plt.xlabel("测试集样本序号", fontsize=12)
plt.ylabel("风速 (m/s)", fontsize=12)
plt.legend(ncol=5, loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(root_path, "多步预测A结果对比图.png"), dpi=300, bbox_inches="tight")
plt.close()

# 8.3 多步预测B对比图
plt.figure(figsize=(16, 7))
plt.plot(x_axis, all_predictions["多步预测B"]["真实值"][:plot_num], label="真实值", color="black", linewidth=1.5)
plt.plot(x_axis, all_predictions["多步预测B"]["线性回归"][:plot_num], label="线性回归", alpha=0.8)
plt.plot(x_axis, all_predictions["多步预测B"]["随机森林"][:plot_num], label="随机森林", alpha=0.8)
plt.plot(x_axis, all_predictions["多步预测B"]["LSTM"][:plot_num], label="LSTM", alpha=0.8)
plt.plot(x_axis, all_predictions["多步预测B"]["Transformer"][:plot_num], label="Transformer", alpha=0.8)
plt.title("多步预测B（未来16小时）：真实值与多模型预测值对比（测试集）", fontsize=14, pad=15)
plt.xlabel("测试集样本序号", fontsize=12)
plt.ylabel("风速 (m/s)", fontsize=12)
plt.legend(ncol=5, loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(root_path, "多步预测B结果对比图.png"), dpi=300, bbox_inches="tight")
plt.close()

# 8.4 RMSE对比柱状图
plt.figure(figsize=(10, 6))
model_names = list(all_results["单步预测"].keys())
rmse_values = [all_results["单步预测"][m]["RMSE"] for m in model_names]
bars = plt.bar(model_names, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=11)
plt.title("单步预测各模型RMSE指标对比", fontsize=14, pad=15)
plt.ylabel("RMSE (m/s)", fontsize=12)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(root_path, "模型RMSE对比柱状图.png"), dpi=300, bbox_inches="tight")
plt.close()

# 8.5 R²对比热力图
r2_matrix = pd.DataFrame(all_results).applymap(lambda x: x["R²"]).T
plt.figure(figsize=(8, 5))
sns.heatmap(r2_matrix, annot=True, fmt='.3f', cmap='YlGnBu', vmin=-1, vmax=1)
plt.title("三类预测任务各模型R²得分对比", fontsize=14, pad=15)
plt.ylabel("预测任务", fontsize=12)
plt.xlabel("模型", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(root_path, "模型R²对比热力图.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\n所有可视化图表已生成并保存到数据集文件夹，共5张：")
print("1. 单步预测结果对比图.png")
print("2. 多步预测A结果对比图.png")
print("3. 多步预测B结果对比图.png")
print("4. 模型RMSE对比柱状图.png")
print("5. 模型R²对比热力图.png")
print("\n所有模型参数已按.pth格式保存，共12个模型文件（4个模型×3类任务）")
print("="*70)
print("全部任务执行完成！")