import requests
from difflib import SequenceMatcher

def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def query_openalex_by_title_and_year(title: str, year: int, email: str):
    """
    用 标题 + 发行年份 查询 OpenAlex
    优先使用 title.search + publication_year 过滤，
    再根据标题相似度选择最佳结果。
    """
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"title.search:{title},publication_year:{year}",
        "per-page": 10,
        "mailto": email
    }
    headers = {
        "User-Agent": f"mailto:{email}"
    }

    print(f"🔍 正在请求 OpenAlex: {url}")
    print(f"📌 参数: {params}")

    response = requests.get(url, params=params, headers=headers, timeout=30)
    print(f"🌐 HTTP 状态码: {response.status_code}")
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])

    if not results:
        print("❌ 没有找到任何结果。")
        return None

    # 按标题相似度排序，取最优结果
    ranked = sorted(
        results,
        key=lambda work: similarity(title, work.get("display_name", "")),
        reverse=True
    )

    best = ranked[0]
    ids = best.get("ids", {}) or {}

    result = {
        "openalex_id": best.get("id"),
        "title": best.get("display_name"),
        "publication_year": best.get("publication_year"),
        "cited_by_count": best.get("cited_by_count"),
        "doi": ids.get("doi"),
        "type": best.get("type"),
        "authors": [
            (a.get("author") or {}).get("display_name")
            for a in (best.get("authorships") or [])
            if (a.get("author") or {}).get("display_name")
        ]
    }

    return result


if __name__ == "__main__":
    title = "Challenges and Pitfalls of Bayesian Unlearning"
    year = 2022
    email = "1367383990@qq.com"   # 改成你的邮箱

    try:
        result = query_openalex_by_title_and_year(title, year, email)

        if result:
            print("\n✅ 查询成功，最佳匹配结果：")
            for k, v in result.items():
                print(f"{k}: {v}")

    except requests.exceptions.Timeout:
        print("\n❌ 请求超时，请检查网络连接。")
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP 错误: {e}")
    except Exception as e:
        print(f"\n❌ 发生其他错误: {e}")