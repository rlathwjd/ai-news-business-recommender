"""
AI타임스 AI산업 기사 크롤러
- AI타임스 목록 페이지에서 기사 URL 수집
- Supabase articles 테이블에서 기존 URL 중복 확인
- 신규 기사 본문 수집
- Supabase articles 테이블에 신규 기사 저장
"""

import time
import os
import re
import html
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    NoSuchElementException,
    TimeoutException,
)
from webdriver_manager.chrome import ChromeDriverManager

from dotenv import load_dotenv
from db.supabase_client import get_supabase


load_dotenv()

BASE_URL = "https://www.aitimes.com"
LIST_URL = "https://www.aitimes.com/news/articleList.html?sc_multi_code=S2&view_type=sm"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36"
    )
}


def normalize_article_url(href: str) -> str:
    """
    AI타임스 기사 URL을 articleView의 idxno 기준으로 정규화합니다.
    같은 기사인데 URL 형태가 달라 중복으로 잡히는 문제를 줄입니다.
    """
    full_url = urljoin(BASE_URL, href)

    parsed = urlparse(full_url)
    query = parse_qs(parsed.query)

    idxno = query.get("idxno", [None])[0]

    if idxno:
        return f"{BASE_URL}/news/articleView.html?idxno={idxno}"

    return full_url.split("#")[0]


def create_driver() -> webdriver.Chrome:
    """Selenium Chrome Driver 생성"""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )


def extract_title_from_item(item) -> str:
    """
    기사 카드에서 제목 후보를 여러 방식으로 추출합니다.
    제목이 없어도 신규 URL이면 버리지 않도록 마지막에는 '제목 없음'을 반환합니다.
    """
    title_candidates = []

    article_links = item.select("a[href*='articleView']")

    for link in article_links:
        text = link.get_text(" ", strip=True)
        if text:
            title_candidates.append(text)

        img = link.select_one("img")
        if img:
            alt = img.get("alt", "").strip()
            if alt:
                title_candidates.append(alt)

    title_tag = item.select_one(
        "h2, h3, strong, .altlist-title, .altlist-titles, .titles"
    )
    if title_tag:
        text = title_tag.get_text(" ", strip=True)
        if text:
            title_candidates.append(text)

    if not title_candidates:
        return "제목 없음"

    title = max(title_candidates, key=len)
    return html.unescape(title).strip() or "제목 없음"


def get_existing_urls() -> set[str]:
    """
    Supabase articles 테이블에서 기존 URL 전체를 가져옵니다.
    URL은 크롤링 URL과 같은 기준으로 정규화합니다.
    """
    supabase = get_supabase()

    urls = set()
    page_size = 1000
    start = 0

    while True:
        result = (
            supabase.table("articles")
            .select("url")
            .range(start, start + page_size - 1)
            .execute()
        )

        rows = result.data or []

        for row in rows:
            url = row.get("url")
            if url:
                urls.add(normalize_article_url(url))

        if len(rows) < page_size:
            break

        start += page_size

    return urls


def get_article_urls(existing_urls: set[str], max_clicks: int | None = 30) -> list[dict]:
    """
    AI타임스 목록 페이지에서 신규 기사 URL만 수집합니다.

    기준:
    - existing_urls: Supabase에 이미 저장된 URL
    - seen: 이번 크롤링 실행 중 이미 확인한 URL
    - DB에도 없고 seen에도 없으면 신규 기사
    """
    print("[크롤러] AI타임스 목록 페이지 접속 중...")

    driver = create_driver()
    driver.get(LIST_URL)
    time.sleep(2)

    all_articles = []
    seen = set()

    def extract_articles():
        soup = BeautifulSoup(driver.page_source, "html.parser")
        items = soup.select("ul.altlist-webzine li.altlist-webzine-item")

        articles = []

        total_items = 0
        existing_count = 0
        seen_count = 0
        new_count = 0
        skipped_count = 0

        for item in items:
            total_items += 1

            link = (
                item.select_one("a.altlist-image[href*='articleView']")
                or item.select_one("a[href*='articleView']")
            )

            if not link:
                skipped_count += 1
                continue

            href = link.get("href", "")
            if not href:
                skipped_count += 1
                continue

            full_url = normalize_article_url(href)

            # 이번 실행에서 이미 처리한 URL이면 중복
            if full_url in seen:
                seen_count += 1
                continue

            # Supabase에 이미 있는 URL이면 기존 기사
            if full_url in existing_urls:
                seen.add(full_url)
                existing_count += 1
                continue

            # 여기까지 왔으면 신규 기사
            title = extract_title_from_item(item)

            seen.add(full_url)
            new_count += 1

            articles.append(
                {
                    "url": full_url,
                    "title": title,
                }
            )

        stats = {
            "total_items": total_items,
            "existing": existing_count,
            "seen": seen_count,
            "new": new_count,
            "skipped": skipped_count,
        }

        return articles, stats

    try:
        first_articles, stats = extract_articles()
        all_articles.extend(first_articles)

        print(
            f"  → 초기 신규 기사 {len(first_articles)}개 수집 "
            f"(전체 기사카드 {stats['total_items']} / 기존 {stats['existing']} / "
            f"중복 {stats['seen']} / 제외 {stats['skipped']})"
        )

        click_count = 0

        while True:
            if max_clicks is not None and click_count >= max_clicks:
                print(f"  → 최대 더보기 클릭 수 {max_clicks}회 도달. 종료")
                break

            try:
                before_count = len(
                    driver.find_elements(
                        By.CSS_SELECTOR,
                        "ul.altlist-webzine li.altlist-webzine-item",
                    )
                )

                more_button = driver.find_element(
                    By.CSS_SELECTOR,
                    "button.list-btn-more",
                )

                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});",
                    more_button,
                )

                time.sleep(1)
                driver.execute_script("arguments[0].click();", more_button)

                WebDriverWait(driver, 10).until(
                    lambda d: len(
                        d.find_elements(
                            By.CSS_SELECTOR,
                            "ul.altlist-webzine li.altlist-webzine-item",
                        )
                    )
                    > before_count
                )

                click_count += 1

                new_articles, stats = extract_articles()
                all_articles.extend(new_articles)

                print(
                    f"  → 더보기 {click_count}회 클릭: "
                    f"신규 {len(new_articles)}개 / 누적 {len(all_articles)}개 "
                    f"(전체 기사카드 {stats['total_items']} / 기존 {stats['existing']} / "
                    f"중복 {stats['seen']} / 제외 {stats['skipped']})"
                )

            except UnexpectedAlertPresentException:
                try:
                    alert = driver.switch_to.alert
                    print(f"  → 더보기 종료: 사이트 알림 - {alert.text}")
                    alert.accept()
                except Exception:
                    print("  → 더보기 종료: 사이트 알림 발생")
                break

            except (NoSuchElementException, TimeoutException):
                print("  → 더보기 버튼이 없거나 추가 기사 로딩이 없어 종료")
                break

            except Exception as e:
                print(f"  → 더보기 종료 또는 실패: {e}")
                break

    finally:
        driver.quit()

    print(f"  → 총 {len(all_articles)}개 신규 기사 URL 추출")
    return all_articles


def extract_published_date(soup: BeautifulSoup) -> str:
    """
    기사 입력 날짜 추출
    반환 예: 2026-05-06 13:58
    """
    published_date = "날짜 미상"

    date_tag = soup.select_one("div.info-update-origin")

    if date_tag:
        raw_text = date_tag.get_text(" ", strip=True)
    else:
        raw_text = ""

        li_tags = soup.select("ul.breadcrumbs li")

        for li in li_tags:
            text = li.get_text(" ", strip=True)

            if "입력" in text:
                raw_text = text
                break

    if raw_text:
        match = re.search(
            r"20\d{2}[.\-/]\d{2}[.\-/]\d{2}\s+\d{2}:\d{2}",
            raw_text,
        )

        if match:
            published_date = (
                match.group(0)
                .replace(".", "-")
                .replace("/", "-")
            )

    return published_date


def extract_title_from_article_page(soup: BeautifulSoup, fallback_title: str) -> str:
    """
    목록에서 제목을 못 가져온 경우 기사 상세 페이지에서 제목을 보완합니다.
    """
    title_tag = (
        soup.select_one("h1.heading")
        or soup.select_one("h1.article-head-title")
        or soup.select_one("h1")
        or soup.select_one("meta[property='og:title']")
    )

    if title_tag:
        if title_tag.name == "meta":
            title = title_tag.get("content", "").strip()
        else:
            title = title_tag.get_text(" ", strip=True)

        title = html.unescape(title).strip()

        if title:
            return title

    return fallback_title


def scrape_article(url: str, title: str) -> dict | None:
    """
    개별 기사 본문 크롤링
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.encoding = "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")

        # 상세 페이지에서 제목 보완
        title = extract_title_from_article_page(soup, title)

        content_area = (
            soup.select_one("div#article-view-content-div")
            or soup.select_one("div.article-body")
        )

        if not content_area:
            print(f"  ⚠ 본문 못 찾음: {title[:30]}")
            return None

        for tag in content_area.select("script, style, figure, .ad, iframe"):
            tag.decompose()

        body_text = content_area.get_text(separator="\n", strip=True)

        if len(body_text) < 200:
            print(f"  ⚠ 본문 너무 짧음: {title[:30]}")
            return None

        published_date = extract_published_date(soup)

        return {
            "url": url,
            "title": title,
            "body": body_text,
            "published_date": published_date,
            "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        print(f"  ❌ 오류 ({title[:30]}): {e}")
        return None


def save_articles_to_db(articles: list[dict]):
    """
    신규 기사 목록을 Supabase articles 테이블에 저장합니다.
    url 컬럼에 UNIQUE 제약조건이 있어야 중복 저장이 방지됩니다.
    """
    if not articles:
        print("  → 저장할 신규 기사 없음")
        return

    supabase = get_supabase()

    payload = []

    for article in articles:
        payload.append(
            {
                "url": article["url"],
                "title": article["title"],
                "body": article["body"],
                "published_date": article["published_date"],
                "crawled_at": article["crawled_at"],
                "embedded": False,
            }
        )

    try:
        result = (
            supabase.table("articles")
            .upsert(
                payload,
                on_conflict="url",
                ignore_duplicates=True,
            )
            .execute()
        )

        saved_count = len(result.data or [])

        print(f"  → Supabase 저장 요청 {len(payload)}개")
        print(f"  → Supabase 신규 저장 {saved_count}개")

    except Exception as e:
        print(f"  ❌ Supabase 저장 실패: {e}")
        raise


def crawl(max_clicks: int | None = 30) -> list[dict]:
    print("=" * 40)
    print("AI타임스 크롤링 시작")
    print("=" * 40)

    existing_urls = get_existing_urls()
    print(f"[크롤러] Supabase 기존 기사 수: {len(existing_urls)}개")

    article_list = get_article_urls(
        existing_urls=existing_urls,
        max_clicks=max_clicks,
    )

    print("\n[크롤러] 기사 본문 수집 중...")

    new_results = []

    for i, article in enumerate(article_list):
        print(
            f"  ({i + 1}/{len(article_list)}) "
            f"{article['title'][:40]}"
        )

        data = scrape_article(
            article["url"],
            article["title"],
        )

        if data:
            new_results.append(data)

        time.sleep(0.5)

    save_articles_to_db(new_results)

    print(f"\n✅ 신규 {len(new_results)}개 기사 본문 수집 완료")

    return new_results


if __name__ == "__main__":
    crawl(10)