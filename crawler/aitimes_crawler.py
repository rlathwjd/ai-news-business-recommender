"""
AI타임스 AI산업 기사 크롤러
"""

import time
import json
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.aitimes.com"
LIST_URL = "https://www.aitimes.com/news/articleList.html?sc_multi_code=S2&view_type=sm"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def get_article_urls(max_articles: int = 20) -> list[dict]:
    """목록 페이지에서 기사 URL 추출"""
    print(f"[크롤러] 목록 페이지 수집 중...")

    response = requests.get(LIST_URL, headers=HEADERS, timeout=10)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    seen = set()

    links = soup.select("a[href*='articleView']")
    for link in links:
        href = link.get("href", "")
        if not href or href in seen:
            continue
        seen.add(href)

        title = link.get_text(strip=True)
        if not title:
            continue

        full_url = BASE_URL + href if href.startswith("/") else href
        articles.append({"url": full_url, "title": title})

        if len(articles) >= max_articles:
            break

    print(f"  → {len(articles)}개 기사 URL 추출")
    return articles


def scrape_article(url: str, title: str) -> dict | None:
    """개별 기사 본문 크롤링"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        # 본문 추출
        content_area = (
            soup.select_one("div#article-view-content-div")
            or soup.select_one("div.article-body")
        )
        if not content_area:
            print(f"  ⚠ 본문 못 찾음: {title[:30]}")
            return None

        # 불필요한 태그 제거
        for tag in content_area.select("script, style, figure, .ad, iframe"):
            tag.decompose()

        body_text = content_area.get_text(separator="\n", strip=True)

        if len(body_text) < 200:
            print(f"  ⚠ 본문 너무 짧음: {title[:30]}")
            return None

        # 날짜 추출
        date_tag = soup.select_one("em.info-data") or soup.select_one("span.date")
        published_date = date_tag.get_text(strip=True) if date_tag else "날짜 미상"

        return {
            "url": url,
            "title": title,
            "body": body_text,
            "published_date": published_date,
            "crawled_at": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"  ❌ 오류 ({title[:30]}): {e}")
        return None


def crawl(max_articles: int = 20, save_path: str = "./data/raw_articles.json") -> list[dict]:
    """전체 크롤링 실행 + JSON 저장"""
    print("=" * 40)
    print("AI타임스 크롤링 시작")
    print("=" * 40)

    # 기사 URL 수집
    article_list = get_article_urls(max_articles)

    # 본문 크롤링
    print(f"\n[크롤러] 기사 본문 수집 중...")
    results = []
    for i, article in enumerate(article_list):
        print(f"  ({i+1}/{len(article_list)}) {article['title'][:40]}")
        data = scrape_article(article["url"], article["title"])
        if data:
            results.append(data)
        time.sleep(0.5)  # 서버 부하 방지

    # JSON 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(results)}개 기사 수집 완료 → {save_path}")
    return results


if __name__ == "__main__":
    crawl()