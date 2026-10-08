import pandas as pd
import requests
import time
from tqdm import tqdm
from pathlib import Path
import logging

LOG_PATH = Path("logs/collect_citations.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EMAIL = "1367383990@qq.com"
OPENALEX_HEADERS = {"User-Agent": f"mailto:{EMAIL}"} if EMAIL else {}

MAX_REQUEST_RETRIES = 3
RETRY_BACKOFF = 5
OPENALEX_BASE_URL = "https://api.openalex.org/works"


def normalize_arxiv_id(arxiv_id: str) -> str:
    clean_id = str(arxiv_id).strip()
    if clean_id.count("v") == 1:
        base, version = clean_id.rsplit("v", 1)
        if version.isdigit():
            return base
    return clean_id


def build_arxiv_doi(arxiv_id: str) -> str:
    clean_id = normalize_arxiv_id(arxiv_id)
    return f"10.48550/arXiv.{clean_id}"


def openalex_get(url, params=None, timeout=20):
    if params is None:
        params = {}

    if EMAIL:
        params = {**params, "mailto": EMAIL}

    last_exception = None
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=OPENALEX_HEADERS, timeout=timeout)

            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                return response
            elif response.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    "OpenAlex status=%s on %s attempt %d/%d, params=%s",
                    response.status_code, url, attempt, MAX_REQUEST_RETRIES, params
                )
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                logger.warning(
                    "OpenAlex unexpected status=%s url=%s params=%s body=%s",
                    response.status_code, url, params, response.text[:500]
                )
                return response
        except Exception as e:
            last_exception = e
            logger.warning(
                "OpenAlex request error on %s attempt %d/%d: %s",
                url, attempt, MAX_REQUEST_RETRIES, e
            )
            time.sleep(RETRY_BACKOFF * attempt)

    if last_exception:
        raise last_exception
    return None


def get_paper_info_by_arxiv(arxiv_id):
    """
    通过 arXiv ID 构造 DOI，然后去 OpenAlex 查询目标论文信息：
    - cited_by_count
    - publication_year
    - publication_date
    - title
    """
    clean_id = normalize_arxiv_id(arxiv_id)
    doi = build_arxiv_doi(clean_id)
    url = f"{OPENALEX_BASE_URL}/doi:{doi}"

    try:
        response = openalex_get(url, timeout=20)
        if response is not None and response.status_code == 200:
            work = response.json()
            return {
                "citation_count": work.get("cited_by_count", 0),
                "publication_year": work.get("publication_year"),
                "publication_date": work.get("publication_date"),
                "title_from_openalex": work.get("display_name", "") or work.get("title", ""),
                "doi": doi,
                "openalex_id": work.get("id"),
                "type": work.get("type"),
            }

        if response is not None and response.status_code == 404:
            logger.warning("OpenAlex DOI lookup not found for arXiv ID %s -> DOI %s", arxiv_id, doi)
            return None

    except Exception as e:
        logger.warning(f"OpenAlex fetch error for {arxiv_id}: {e}")

    logger.error("Exceeded max retries for %s", clean_id)
    return None


def collect_citations(csv_path, output_path, sleep_seconds=1.0, resume=True):
    input_file = Path(csv_path)
    output_file = Path(output_path)

    df = pd.read_csv(input_file)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    if "arxiv_id" not in df.columns:
        raise ValueError("输入数据缺少 arxiv_id 列")

    processed_ids = set()
    results = []

    # 断点续跑
    if resume and output_file.exists():
        existing_df = pd.read_csv(output_file)
        if "arxiv_id" in existing_df.columns:
            existing_df["arxiv_id"] = existing_df["arxiv_id"].astype(str)
            processed_ids = set(existing_df["arxiv_id"])
            results = existing_df.to_dict("records")
            logger.info("Resume enabled, loaded %d existing rows from %s", len(existing_df), output_path)

    print("开始收集 citation 数据...")

    success_count = 0
    skip_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        arxiv_id = str(row["arxiv_id"]).strip()

        if not arxiv_id:
            logger.warning("Skipping row with empty arxiv_id")
            skip_count += 1
            continue

        if arxiv_id in processed_ids:
            continue

        paper_info = get_paper_info_by_arxiv(arxiv_id)

        # 查不到：不写入新数据集
        if paper_info is None:
            logger.warning("Skipping %s because paper info not found.", arxiv_id)
            skip_count += 1
            time.sleep(sleep_seconds)
            continue

        result_row = row.to_dict()
        result_row.update({
            "citation_count": paper_info["citation_count"],
            "publication_year": paper_info["publication_year"],
            "publication_date": paper_info["publication_date"],
            "title_from_openalex": paper_info["title_from_openalex"],
            "openalex_id": paper_info["openalex_id"],
            "constructed_doi": paper_info["doi"],
            "openalex_type": paper_info["type"],
        })

        results.append(result_row)
        processed_ids.add(arxiv_id)
        success_count += 1

        logger.info(
            "Collected %s | citation_count=%s | publication_year=%s | publication_date=%s",
            arxiv_id,
            paper_info["citation_count"],
            paper_info["publication_year"],
            paper_info["publication_date"],
        )

        # 每 50 条写入一次
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)
            logger.info("Checkpoint saved to %s (%d rows).", output_file, len(results))
            print(f"已写入 checkpoint: {len(results)} 条")

        time.sleep(sleep_seconds)

    pd.DataFrame(results).to_csv(output_file, index=False)
    logger.info(
        "Finished. Saved %d valid rows to %s. skipped=%d",
        len(results), output_file, skip_count
    )

    print(f"处理完成！成功保存 {len(results)} 条，跳过 {skip_count} 条。")
    print(f"结果文件：{output_file}")


def main():
    collect_citations(
        csv_path="/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata_2022_keywords.csv",
        output_path="/Users/ruyi/Desktop/NAIP/dataset/arxiv_cs_metadata_2022_with_citations.csv",
        sleep_seconds=1.0,
        resume=True,
    )


if __name__ == "__main__":
    main()