import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ====== 路径配置 ======
csv_path = "/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata_2022_with_citations.csv"
output_dir = Path("/Users/ruyi/Desktop/NAIP/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ====== 读取数据 ======
df = pd.read_csv(csv_path)

if "citation_count" not in df.columns:
    raise ValueError("CSV 中不存在 citation_count 列")

# 转成数值
df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce")

# 去掉缺失和负值
cite = df["citation_count"].dropna()
cite = cite[cite >= 0]

if len(cite) == 0:
    raise ValueError("citation_count 列没有可用数据")

# 对数变换
log_cite = np.log1p(cite)

# ====== 统计信息 ======
print("citation_count 描述统计：")
print(cite.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

# ====== 可视化 ======
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始直方图
axes[0].hist(cite, bins=50, edgecolor="black", color="skyblue")
axes[0].set_title("Citation Count Distribution")
axes[0].set_xlabel("citation_count")
axes[0].set_ylabel("Frequency")

# 对数直方图
axes[1].hist(log_cite, bins=50, edgecolor="black", color="salmon")
axes[1].set_title("Log(1 + Citation Count) Distribution")
axes[1].set_xlabel("log(1 + citation_count)")
axes[1].set_ylabel("Frequency")

# 箱线图
axes[2].boxplot(cite, vert=True)
axes[2].set_title("Citation Count Boxplot")
axes[2].set_ylabel("citation_count")

plt.tight_layout()

# 保存
save_path = output_dir / "citation_count_distribution.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"图像已保存到: {save_path}")