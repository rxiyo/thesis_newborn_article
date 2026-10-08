import time
import logging
from pathlib import Path

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from tqdm import tqdm


# =========================
# 基础配置
# =========================
EMAIL = "1367383990@qq.com"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_CONCEPTS_URL = "https://api.openalex.org/concepts"

MAX_REQUEST_RETRIES = 3
RETRY_BACKOFF = 5
BATCH_SIZE = 200  # OpenAlex works 单页最大 200

# 只检索 CS concept
CONCEPT_QUERY = "Computer Science"

DATE_RANGE = ("2022-01-01", "2024-12-31")
CITED_BY_THRESHOLD = 50
PAPERS_LIMIT = 3000

OUTPUT_PATH = Path("/Users/ruyi/Desktop/NAIP/dataset/high_cited_papers.csv")
LOG_FILE = Path("/Users/ruyi/Desktop/NAIP/logs/high_cited_papers.log")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_maxsize=10))
SESSION.mount("http://", HTTPAdapter(pool_maxsize=10))
SESSION.headers.update({
    "User-Agent": f"OpenAlexClient/1.0 (mailto:{EMAIL})"
})

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =========================
# 通用请求函数
# =========================
def get_json(url, params=None, timeout=30):
    """带重试的 GET 请求"""
    params = params or {}
    if EMAIL:
        params["mailto"] = EMAIL

    last_exception = None

    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                logger.warning(f"429 限流：{url}，第 {attempt}/{MAX_REQUEST_RETRIES} 次重试")
                time.sleep(RETRY_BACKOFF * attempt)
                continue

            logger.warning(
                f"请求失败：status={resp.status_code}, url={resp.url}, body={resp.text[:500]}"
            )
            return None

        except Exception as e:
            last_exception = e
            logger.warning(f"请求异常：{url}，第 {attempt}/{MAX_REQUEST_RETRIES} 次重试，错误：{e}")
            time.sleep(RETRY_BACKOFF * attempt)

    logger.error(f"请求最终失败：{url}，最后异常：{last_exception}")
    return None


# =========================
# Concept 解析
# =========================
def search_best_concept(query):
    """
    根据关键词搜索最合适的 concept
    返回:
        {
            "id": "https://openalex.org/C....",
            "short_id": "C....",
            "display_name": "...",
            "level": ...,
            "works_count": ...
        }
    """
    params = {
        "search": query,
        "per-page": 10,
        "select": "id,display_name,level,works_count"
    }
    data = get_json(OPENALEX_CONCEPTS_URL, params=params)
    if not data or not data.get("results"):
        logger.error(f"未搜索到 concept：{query}")
        return None

    results = data["results"]
    q = query.strip().lower()

    exact_candidates = [
        x for x in results
        if x.get("display_name", "").strip().lower() == q
    ]
    if exact_candidates:
        best = sorted(exact_candidates, key=lambda x: x.get("works_count", 0), reverse=True)[0]
    else:
        contains_candidates = [
            x for x in results
            if q in x.get("display_name", "").strip().lower()
        ]
        if contains_candidates:
            best = sorted(contains_candidates, key=lambda x: x.get("works_count", 0), reverse=True)[0]
        else:
            best = sorted(results, key=lambda x: x.get("works_count", 0), reverse=True)[0]

    concept_id_full = best["id"]
    concept_id_short = concept_id_full.rstrip("/").split("/")[-1]

    logger.info(
        f"concept 解析成功：query='{query}' -> {best.get('display_name')} "
        f"({concept_id_short}), level={best.get('level')}, works_count={best.get('works_count')}"
    )

    return {
        "id": concept_id_full,
        "short_id": concept_id_short,
        "display_name": best.get("display_name"),
        "level": best.get("level"),
        "works_count": best.get("works_count"),
    }


# =========================
# Abstract 反解码
# =========================
def reconstruct_abstract(inverted_index):
    """
    OpenAlex 的 abstract_inverted_index 转回正常 abstract 文本
    """
    if not inverted_index:
        return None

    try:
        position_word = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word[pos] = word

        if not position_word:
            return None

        max_pos = max(position_word.keys())
        words = [position_word.get(i, "") for i in range(max_pos + 1)]
        abstract = " ".join(words).strip()
        return abstract if abstract else None
    except Exception as e:
        logger.warning(f"abstract 重建失败：{e}")
        return None


# =========================
# 字段提取
# =========================
def extract_arxiv_id(ids):
    arxiv_id = (ids or {}).get("arxiv")
    if arxiv_id:
        return arxiv_id.rstrip("/").split("/")[-1]
    return None


def extract_category(item):
    """
    category 优先取 primary_topic.display_name
    """
    primary_topic = item.get("primary_topic") or {}
    return primary_topic.get("display_name")


# =========================
# 按 concept 检索 works
# =========================
def fetch_top_papers_for_concept(concept_id_short, concept_name, start_date, end_date,
                                 cited_by_threshold, max_results):
    """
    按 concepts.id 抓取高被引论文
    """
    results = []
    next_cursor = "*"
    total_fetched = 0

    logger.info(f"开始检索 concept={concept_name} ({concept_id_short})，目标总数：{max_results}")

    with tqdm(total=max_results, desc=f"{concept_name} 进度") as pbar:
        while len(results) < max_results:
            params = {
                "filter": ",".join([
                    f"concepts.id:https://openalex.org/{concept_id_short}",
                    f"from_publication_date:{start_date}",
                    f"to_publication_date:{end_date}",
                    f"cited_by_count:>{cited_by_threshold}",
                ]),
                "sort": "cited_by_count:asc",
                "per-page": min(BATCH_SIZE, max_results - len(results)),
                "cursor": next_cursor,
                "select": ",".join([
                    "id",
                    "display_name",
                    "publication_date",
                    "cited_by_count",
                    "doi",
                    "ids",
                    "abstract_inverted_index",
                    "primary_topic",
                ]),
            }

            data = get_json(OPENALEX_WORKS_URL, params=params, timeout=40)
            if not data:
                logger.warning(f"concept={concept_name} 请求失败，提前结束")
                break

            batch = data.get("results", [])
            new_records = len(batch)

            if new_records == 0:
                logger.warning(
                    f"concept={concept_name} 本页返回 0 条。"
                    f" cursor={next_cursor}, filter={params['filter']}"
                )
                break

            results.extend(batch)
            total_fetched += new_records
            pbar.update(new_records)

            next_cursor = data.get("meta", {}).get("next_cursor")
            logger.info(
                f"concept={concept_name}：成功获取 {new_records} 篇，总计 {total_fetched} 篇"
            )

            if not next_cursor:
                break

            time.sleep(1)

    return results[:max_results]


# =========================
# 数据清洗与导出
# =========================
def save_to_csv(data, output_path):
    if not data:
        logger.warning("未检索到任何数据，无法保存 CSV 文件。")
        return

    rows = []
    for item in data:
        rows.append({
            "arxiv_id": extract_arxiv_id(item.get("ids", {})),
            "doi": item.get("doi"),
            "category": extract_category(item),
            "title": item.get("display_name"),
            "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
            "published": item.get("publication_date"),
            "citation_count": item.get("cited_by_count"),
        })

    df = pd.DataFrame(rows)

    # 去重：优先 arxiv_id，再 doi，再 title+published
    if not df.empty:
        df_with_arxiv = df[df["arxiv_id"].notna()].drop_duplicates(subset=["arxiv_id"], keep="first")
        df_without_arxiv = df[df["arxiv_id"].isna()]

        df_with_doi = df_without_arxiv[df_without_arxiv["doi"].notna()].drop_duplicates(subset=["doi"], keep="first")
        df_without_doi = df_without_arxiv[df_without_arxiv["doi"].isna()]

        df_without_doi = df_without_doi.drop_duplicates(subset=["title", "published"], keep="first")

        df = pd.concat([df_with_arxiv, df_with_doi, df_without_doi], ignore_index=True)

    # df.to_csv(output_path, index=False, encoding="utf-8-sig")
    df.to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
        encoding="utf-8-sig"
    )
    logger.info(f"已保存 {len(df)} 条结果到 {output_path}")


# =========================
# 主流程
# =========================
def main():
    logger.info("开始高被引论文检索...")

    concept_info = search_best_concept(CONCEPT_QUERY)
    if concept_info is None:
        logger.error("concept 解析失败，程序结束")
        return

    results = fetch_top_papers_for_concept(
        concept_id_short=concept_info["short_id"],
        concept_name=concept_info["display_name"],
        start_date=DATE_RANGE[0],
        end_date=DATE_RANGE[1],
        cited_by_threshold=CITED_BY_THRESHOLD,
        max_results=PAPERS_LIMIT,
    )

    logger.info(f"{concept_info['display_name']}：共检索到 {len(results)} 篇符合条件的论文")

    save_to_csv(results, OUTPUT_PATH)
    logger.info("检索完成！")


if __name__ == "__main__":
    main()