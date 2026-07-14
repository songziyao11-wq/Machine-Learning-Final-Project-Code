import pandas as pd
import os

# 1. 配置路径与文件参数
# 数据集根目录（你的文件夹路径）
root_path = r"C:\Users\23768\Desktop\机器学习数据集"
# 三个高度对应的子文件夹名
height_folders = ["10m_data", "50m_data", "100m_data"]
# 每个子文件夹下的parquet文件列表
parquet_file_list = [
    "train-00000-of-00001.parquet",
    "val-00000-of-00001.parquet",
    "test-00000-of-00001.parquet"
]

# 2. 定义单高度数据读取函数
def load_single_height_data(folder_path, height_tag):
    """
    读取单个高度的train/val/test文件，合并为完整数据集，并重命名字段
    :param folder_path: 对应高度的文件夹路径
    :param height_tag: 高度标识（10m/50m/100m），用于区分字段
    :return: 处理后的单高度数据集
    """
    # 读取并合并该高度下所有parquet文件
    df_list = []
    for file_name in parquet_file_list:
        full_file_path = os.path.join(folder_path, file_name)
        df = pd.read_parquet(full_file_path)
        df_list.append(df)
    height_full_df = pd.concat(df_list, ignore_index=True)

    # 按照作业要求的字段规范重命名，修正原始数据的拼写错误
    rename_map = {
        "Date & Time Stamp": "Timestamp",
        "SpeedAvg": f"WindSpeed_{height_tag}",
        "DirectionAvg": f"WindDirection_{height_tag}",
        "TemperatureAvg": f"Temperature_{height_tag}",
        "PressureAvg": f"Pressure_{height_tag}",
        "HumidtyAvg": f"Humidity_{height_tag}"
    }
    height_full_df = height_full_df.rename(columns=rename_map)

    # 仅保留作业要求的核心字段，剔除冗余列
    keep_columns = ["Timestamp"] + [
        f"WindSpeed_{height_tag}",
        f"WindDirection_{height_tag}",
        f"Temperature_{height_tag}",
        f"Pressure_{height_tag}",
        f"Humidity_{height_tag}"
    ]
    height_full_df = height_full_df[keep_columns]

    # 时间戳转为标准时间格式，方便后续排序与时序处理
    height_full_df["Timestamp"] = pd.to_datetime(height_full_df["Timestamp"])
    return height_full_df

# 3. 读取三个高度的数据集
df_10m = load_single_height_data(os.path.join(root_path, "10m_data"), "10m")
df_50m = load_single_height_data(os.path.join(root_path, "50m_data"), "50m")
df_100m = load_single_height_data(os.path.join(root_path, "100m_data"), "100m")

# 4. 按时间戳横向拼接三个高度的数据集
# 以时间戳为基准，保留三个高度都有数据的时间点，保证数据完整性
merged_data = df_10m.merge(df_50m, on="Timestamp", how="inner")
merged_data = merged_data.merge(df_100m, on="Timestamp", how="inner")

# 按时间升序排序，符合时间序列数据的规范
merged_data = merged_data.sort_values("Timestamp").reset_index(drop=True)

# 5. 保存整合后的总数据集
save_path = os.path.join(root_path, "merged_wind_dataset.csv")
merged_data.to_csv(save_path, index=False, encoding="utf-8-sig")

# 6. 输出结果提示
print(f"数据拼接完成！总数据量：{len(merged_data)} 条")
print(f"整合后的数据集已保存至：{save_path}")
print("\n整合后的完整字段列表：")
print(merged_data.columns.tolist())