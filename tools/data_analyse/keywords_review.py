import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
from tqdm import tqdm  # 引入 tqdm 进度条库
from langchain_core.messages import HumanMessage, SystemMessage
from tools.poe_chat_client import PoeChatOpenAI as ChatOpenAI

original_csv = Path("/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata_keywords.csv")  # 原始预测结果
suspicious_csv = Path("/Users/ruyi/Desktop/NAIP/dataset/suspicious_keywords.csv")  # 可疑关键词文件
output_csv = Path("/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata_keywords_reviewed.csv")  # 审核后的输出
model_name = "gpt-4o"
max_retries = 3
retry_delay = 5.0

system_prompt = (
    "You are an expert metadata auditor tasked with evaluating and fixing automatically generated research keywords. "
    "Your role is to address flagged issues in the suspicious keyword based on the provided heuristic reasons. "
    "Only make changes if necessary to meet the desired standards: concise, canonical research term suitable for indexing academic work. "
    "Keep the response concise, strictly addressing the issues flagged in the reasons. "
    "Respond in JSON format with an action and explanation."
)

user_template = """Paper title: {title}
Abstract: {abstract}

Existing keyword: "{keyword}"
suspicious_reasons: {reasons}

Reply with strict JSON:
{{"action": "keep", "explanation": "short reason"}}
or
{{"action": "replace", "keyword": "new keyword", "explanation": "short reason"}}
"""

base_df = pd.read_csv(original_csv)
suspicious_df = pd.read_csv(suspicious_csv)

required_cols = {"keyword", "suspicious_reasons"}
missing_cols = required_cols - set(suspicious_df.columns)
if missing_cols:
    raise ValueError(f"{suspicious_csv} 缺少必要列: {', '.join(sorted(missing_cols))}")

if "arxiv_id" not in base_df.columns:
    raise ValueError("原始 CSV 中缺少 arxiv_id 列，无法定位行。")

chat = ChatOpenAI(model=model_name, temperature=0.0)
updated_rows = 0
kept_rows = 0
failed_rows = []
modified_records = []

# 使用 tqdm 添加带总数的进度条，传递 total 显示总条目数
with tqdm(total=len(suspicious_df), desc="Processing records", unit="record") as pbar:
    for row in suspicious_df.itertuples(index=False):
        arxiv_id = getattr(row, "arxiv_id", "")
        title = getattr(row, "title", "")
        abstract = getattr(row, "abstract", "")
        keyword = getattr(row, "keyword", "")
        reasons = getattr(row, "suspicious_reasons", "")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=user_template.format(
                    title=title.strip() or "<missing>",
                    abstract=abstract.strip() or "<missing>",
                    keyword=keyword.strip() or "<missing>",
                    reasons=reasons.strip() or "none",
                )
            ),
        ]

        decision = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = chat.batch([messages])[0].content.strip()
                decision = json.loads(raw_response)
                if decision.get("action") not in {"keep", "replace"}:
                    raise ValueError(f"Unexpected action: {decision}")
                if decision["action"] == "replace" and not decision.get("keyword"):
                    raise ValueError("Replacement keyword missing.")
                break
            except Exception as exc:
                if attempt == max_retries:
                    failed_rows.append({
                        "arxiv_id": arxiv_id,
                        "keyword": keyword,
                        "error": str(exc),
                        "raw_response": raw_response if "raw_response" in locals() else "<none>",
                    })
                else:
                    time.sleep(retry_delay * attempt)

        if decision is None:
            pbar.update(1)  # 更新进度条
            continue

        if decision["action"] == "keep":
            kept_rows += 1
            pbar.update(1)  # 更新进度条
            continue

        replacement = decision["keyword"].strip()
        match_mask = base_df["arxiv_id"].astype(str).str.strip() == str(arxiv_id).strip()

        if not match_mask.any():
            failed_rows.append({
                "arxiv_id": arxiv_id,
                "keyword": keyword,
                "error": "No matching arxiv_id in original CSV.",
                "raw_response": decision,
            })
            pbar.update(1)  # 更新进度条
            continue

        base_df.loc[match_mask, "predicted_keyword"] = replacement
        updated_rows += int(match_mask.sum())

        # 收集修改记录
        modified_records.append({
            "arxiv_id": arxiv_id,
            "old_keyword": keyword,
            "new_keyword": replacement,
            "reason": decision.get("explanation", ""),
        })

        pbar.update(1)  # 更新进度条

base_df.to_csv(output_csv, index=False)

if modified_records:
    pd.DataFrame(modified_records).to_csv("modified_keywords.csv", index=False)
    print(f"修改过的记录已保存到: modified_keywords.csv")

if failed_rows:
    failed_df = pd.DataFrame(failed_rows)
    failed_df.to_csv("failed_keywords.csv", index=False)
    print(f"未成功处理的记录已保存到: failed_keywords.csv")


print(f"审核完成：保留 {kept_rows} 条，可疑替换 {updated_rows} 条。")
print(f"结果已写入：{output_csv.resolve()}")
