import pandas as pd

def merge_csv_files(csv_paths, output_path, encoding="utf-8", index=False):
    """合并列结构一致的多个 CSV 文件"""
    dataframes = [pd.read_csv(path, encoding=encoding) for path in csv_paths]
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(output_path, index=index, encoding=encoding)

if __name__ == "__main__":
    csv_files = [
        "/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata_2022.csv",
        "/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata_2023.csv",
        "/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata_2024.csv",
    ]
    merge_csv_files(csv_files, "/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata.csv")