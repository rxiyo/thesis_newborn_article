import math
import time
import json
import logging
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter

# =========================
# 基础与日志配置
# =========================
LOG_PATH = Path("logs/calc_tncsi_tncsisp_all.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EMAIL = "1367383990@qq.com"
OPENALEX_BASE_URL = "https://api.openalex.org/works"

MAX_REQUEST_RETRIES = 3
RETRY_BACKOFF = 5
MIN_HITS = 30
TNCSI_BG_LIMIT = 100
TNCSISP_BG_LIMIT = 100

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_maxsize=10))
SESSION.mount("http://", HTTPAdapter(pool_maxsize=10))
SESSION.headers.update({
    "User-Agent": f"OpenAlexClient/1.0 (mailto:{EMAIL})"
})

TNCSI_CACHE_PATH = Path("dataset/tncsi_background_cache.csv")
TNCSI_PERSISTENT_CACHE = {}

# =========================
# 缓存与工具函数
# =========================
def load_tncsi_persistent_cache():
    """加载 TNCSI 持久化缓存"""
    global TNCSI_PERSISTENT_CACHE
    if not TNCSI_CACHE_PATH.exists():
        TNCSI_PERSISTENT_CACHE = {}
        logger.info("未找到本地 TNCSI 缓存文件，将创建新缓存。")
        return
    try:
        df = pd.read_csv(TNCSI_CACHE_PATH)
        TNCSI_PERSISTENT_CACHE = {
            row["keyword"]: (json.loads(row["citations_json"]), row["hit_count"])
            for _, row in df.iterrows() if pd.notna(row["keyword"])
        }
        logger.info(f"成功加载 TNCSI 本地缓存，共包含 {len(TNCSI_PERSISTENT_CACHE)} 个关键词。")
    except Exception as e:
        logger.warning(f"加载 TNCSI 本地缓存失败: {e}")
        TNCSI_PERSISTENT_CACHE = {}

def save_tncsi_persistent_cache():
    """保存 TNCSI 持久化缓存"""
    try:
        TNCSI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"keyword": keyword, "citations_json": json.dumps(citations), "hit_count": hit_count}
            for keyword, (citations, hit_count) in TNCSI_PERSISTENT_CACHE.items()
        ]
        pd.DataFrame(rows).to_csv(TNCSI_CACHE_PATH, index=False)
        logger.info(f"已将 {len(TNCSI_PERSISTENT_CACHE)} 条 TNCSI 缓存数据保存至 {TNCSI_CACHE_PATH}")
    except Exception as e:
        logger.warning(f"保存 TNCSI 缓存失败: {e}")

def normalize_arxiv_id(arxiv_id: str) -> str:
    """标准化 arXiv ID"""
    clean_id = str(arxiv_id).strip()
    if clean_id.lower().startswith("arxiv:"):
        clean_id = clean_id.split(":", 1)[1]
    if clean_id.count("v") == 1:
        base, version = clean_id.rsplit("v", 1)
        if version.isdigit():
            return base
    return clean_id

def build_arxiv_doi(arxiv_id: str) -> str:
    """构建 arXiv 的 DOI"""
    clean_id = normalize_arxiv_id(arxiv_id)
    return f"10.48550/arXiv.{clean_id}"

# =========================
# OpenAlex 请求逻辑
# =========================
def get_target_citation(arxiv_id: str) -> tuple:
    """
    获取目标论文的引用量。
    返回: (引用量整数或 None, HTTP 状态码)
    """
    doi = build_arxiv_doi(arxiv_id)
    url = f"{OPENALEX_BASE_URL}/doi:{doi}"
    params = {"mailto": EMAIL} if EMAIL else {}

    last_status = 0
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=20)
            last_status = response.status_code
            if response.status_code == 200:
                return response.json().get("cited_by_count", 0), 200
            elif response.status_code == 404:
                logger.warning(f"OpenAlex 找不到该 arXiv ID 的 DOI: {arxiv_id} -> {doi}")
                return None, 404
            elif response.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"请求目标引用量状态码异常 {response.status_code}，正在重试 {attempt}/{MAX_REQUEST_RETRIES}")
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                return None, response.status_code
        except Exception as e:
            logger.warning(f"请求目标引用量发生错误 {arxiv_id} (尝试 {attempt}/{MAX_REQUEST_RETRIES}): {e}")
            time.sleep(RETRY_BACKOFF * attempt)
    return None, last_status

def get_background_citations(query: str, limit: int = 100, start_date: str = None, end_date: str = None) -> tuple:
    """
    获取背景引用数据。
    返回: (引用列表, 命中数量, HTTP 状态码)
    """
    is_tncsi = not start_date and not end_date

    if is_tncsi and query in TNCSI_PERSISTENT_CACHE:
        return TNCSI_PERSISTENT_CACHE[query][0], TNCSI_PERSISTENT_CACHE[query][1], 200

    filters = ["concepts.id:https://openalex.org/C41008148"]
    if start_date and end_date:
        filters.append(f"from_publication_date:{start_date}")
        filters.append(f"to_publication_date:{end_date}")

    params = {
        "search": query,
        "filter": ",".join(filters),
        "select": "cited_by_count",
        "per-page": limit,
    }
    if EMAIL:
        params["mailto"] = EMAIL

    try:
        response = SESSION.get(OPENALEX_BASE_URL, params=params, timeout=20)
        if response.status_code == 200:
            results = response.json().get("results", [])
            citations = [item["cited_by_count"] for item in results if "cited_by_count" in item]
            
            if is_tncsi:
                TNCSI_PERSISTENT_CACHE[query] = (citations, len(citations))
                
            return citations, len(citations), 200
        else:
            logger.warning(f"获取背景引用失败，关键词: {query}, 状态码: {response.status_code}")
            return [], 0, response.status_code
    except Exception as e:
        logger.error(f"请求背景引用异常，关键词: {query}, 错误信息: {e}")
        return [], 0, 0

def calculate_score(target_citation: int, background_citations: list) -> float:
    """计算指标分数"""
    if not background_citations:
        return 0.0
    mean_cite = sum(background_citations) / len(background_citations)
    if mean_cite == 0:
        return 1.0 if target_citation > 0 else 0.5
    z = target_citation / mean_cite
    return round(1 - math.exp(-z), 4)

# =========================
# 主处理流程
# =========================
def process_and_collect(csv_path: str, output_path: str, keyword_column: str, citation_column: str, sleep_seconds: float = 1.0, resume: bool = True):
    input_file = Path(csv_path)
    output_file = Path(output_path)

    load_tncsi_persistent_cache()

    df = pd.read_csv(input_file)
    logger.info(f"从 {input_file} 加载了 {len(df)} 行待处理数据。")

    # if "arxiv_id" not in df.columns:
    #     raise ValueError("输入数据缺少 arxiv_id 列")

    results = []
    processed_ids = set()
    
    # 断点续传逻辑
    # if resume and output_file.exists():
    #     existing_df = pd.read_csv(output_file)
    #     if "arxiv_id" in existing_df.columns:
    #         existing_df["arxiv_id"] = existing_df["arxiv_id"].astype(str)
    #         processed_ids = set(existing_df["arxiv_id"])
    #         logger.info(f"开启断点续传，已加载 {len(existing_df)} 行已处理的数据。")

    if resume and output_file.exists():
        existing_df = pd.read_csv(output_file)
        if "doi" in existing_df.columns:
            existing_df["doi"] = existing_df["doi"].astype(str)
            processed_ids = set(existing_df["doi"])
            logger.info(f"开启断点续传，已加载 {len(existing_df)} 行已处理的数据。")

    print("开始收集引用并计算指标...")
    skip_count = 0
    total_saved = len(processed_ids)
    consecutive_failures = 0  # 连续失败计数器 (仅针对 429)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        row_data = row.to_dict()
        arxiv_id = str(row_data.get("arxiv_id", "")).strip()
        doi=str(row_data.get("doi", "")).strip()

        # if not arxiv_id or arxiv_id in processed_ids:
        #     continue
        if doi in processed_ids:
            continue

        query_keyword = str(row_data.get(keyword_column, "")).strip()
        pub_date_str = str(row_data.get("published", "")).strip()
        target_citation= row_data.get(citation_column, None)

        # 1. 获取目标论文的真实引用量
        # target_citation, target_status = get_target_citation(arxiv_id)
        # if target_citation is None:
        #     logger.warning(f"跳过 {arxiv_id}：无法获取目标论文引用量 (状态码: {target_status})。")
        #     skip_count += 1
            
        #     # 仅在被限流 (429) 时才增加连续失败计数
        #     if target_status == 429:
        #         consecutive_failures += 1
        #         if consecutive_failures >= 5:
        #             logger.error("连续 5 次遇到 429 限流，提前终止程序以保护 API。")
        #             print("\n[警告] 连续 5 次遇到 429 限流，提前终止程序。")
        #             break
        #     else:
        #         # 如果是 404 等正常找不到的情况，清零 429 计数器
        #         consecutive_failures = 0
                
        #     time.sleep(sleep_seconds)
        #     continue

        # row_data["citation_count"] = target_citation

        # 2. 计算时间窗口 (前后180天)
        start_date_str, end_date_str = None, None
        if query_keyword and pub_date_str and pub_date_str != "nan":
            try:
                pub_date = datetime.strptime(pub_date_str[:10], "%Y-%m-%d")
                start_date = pub_date - timedelta(days=180)
                end_date = pub_date + timedelta(days=180)
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
            except ValueError:
                logger.warning(f"论文 {arxiv_id} 的日期格式无效: {pub_date_str}")

        # 3. 计算 TNCSI (使用缓存)
        if query_keyword:
            bg_cites_all, hits_all, bg_status_all = get_background_citations(query_keyword, TNCSI_BG_LIMIT)
            
            if bg_status_all != 200:
                logger.warning(f"跳过 {arxiv_id}：无法获取 TNCSI 背景引用 (状态码: {bg_status_all})。")
                skip_count += 1
                
                if bg_status_all == 429:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        logger.error("连续 5 次请求 TNCSI 背景引用遇到 429 限流，提前终止程序。")
                        print("\n[警告] 连续 5 次请求 TNCSI 背景引用遇到 429 限流，提前终止程序。")
                        break
                else:
                    consecutive_failures = 0
                    
                time.sleep(sleep_seconds)
                continue

            if hits_all >= MIN_HITS:
                row_data["TNCSI"] = calculate_score(target_citation, bg_cites_all)
            else:
                row_data["TNCSI"] = None

            # 4. 计算 TNCSISP (不使用缓存，限制6个月窗口)
            # TNCSISP 关注发表日期前后 6 个月内的同期论文，确保比较的公平性
            if start_date_str and end_date_str:
                bg_cites_sp, hits_sp, bg_status_sp = get_background_citations(
                    query_keyword, 
                    TNCSISP_BG_LIMIT, 
                    start_date=start_date_str, 
                    end_date=end_date_str
                )
                
                if bg_status_sp != 200:
                    logger.warning(f"跳过 {arxiv_id}：无法获取 TNCSISP 背景引用 (状态码: {bg_status_sp})。")
                    skip_count += 1
                    
                    if bg_status_sp == 429:
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            logger.error("连续 5 次请求 TNCSISP 背景引用遇到 429 限流，提前终止程序。")
                            print("\n[警告] 连续 5 次请求 TNCSISP 背景引用遇到 429 限流，提前终止程序。")
                            break
                    else:
                        consecutive_failures = 0
                        
                    time.sleep(sleep_seconds)
                    continue

                if hits_sp >= MIN_HITS:
                    row_data["TNCSISP"] = calculate_score(target_citation, bg_cites_sp)
                else:
                    row_data["TNCSISP"] = None
            else:
                row_data["TNCSISP"] = None
        else:
            row_data["TNCSI"] = None
            row_data["TNCSISP"] = None

        # 如果能顺利走到这里，说明所有需要的 API 请求都成功了，清零失败计数器
        consecutive_failures = 0

        logger.info(
            f"论文: {arxiv_id}, 关键词：{query_keyword}, 引用量: {row_data['citation_count']}, "
            f"TNCSI: {row_data['TNCSI']}, TNCSISP: {row_data['TNCSISP']}"
        )
        
        if row_data["TNCSI"] is not None or row_data["TNCSISP"] is not None:
            results.append(row_data)
            processed_ids.add(arxiv_id)

        # 每处理 50 条保存一次检查点
        if len(results) >= 50:
            header = not output_file.exists()
            pd.DataFrame(results).to_csv(output_file, mode='a', header=header, index=False)
            
            total_saved += len(results)
            save_tncsi_persistent_cache()
            logger.info(f"已保存检查点：本次写入 {len(results)} 条，总计已保存 {total_saved} 条。")
            print(f"\n已追加写入 checkpoint: {len(results)} 条")
            
            results = []

        time.sleep(sleep_seconds)

    # 循环结束后，把剩余不足 50 条的数据写入
    if results:
        header = not output_file.exists()
        pd.DataFrame(results).to_csv(output_file, mode='a', header=header, index=False)
        total_saved += len(results)

    save_tncsi_persistent_cache()
    logger.info(f"处理完成！成功保存 {total_saved} 条，跳过 {skip_count} 条。")
    print(f"\n全部处理完成（或因限制提前终止）！成功保存 {total_saved} 条，跳过 {skip_count} 条。结果文件：{output_file}")

def main():
    process_and_collect(
        csv_path="/Users/ruyi/Desktop/NAIP/dataset/cleaned_high_cited_papers_keywords.csv",
        output_path="/Users/ruyi/Desktop/NAIP/dataset/cleaned_high_cited_papers_keywords_with_scores.csv",
        keyword_column="predicted_keyword",
        citation_column="citation_count",
        sleep_seconds=1.0,
        resume=True,
    )

if __name__ == "__main__":
    main()