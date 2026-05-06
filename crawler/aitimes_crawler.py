"""
AI타임스 AI산업 기사 크롤러
"""

import time
import json
import os
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
import html


BASE_URL = "https://www.aitimes.com"
LIST_URL = "https://www.aitimes.com/news/articleList.html?sc_multi_code=S2&view_type=sm"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def extract_article_urls_from_page(driver, existing_urls, seen):
    soup = BeautifulSoup(driver.page_source, "html.parser")

    articles = []
    found_existing = False

    items = soup.select("ul.altlist-webzine li.altlist-webzine-item")

    for item in items:
        link = item.select_one("a[href*='articleView']")
        if not link:
            continue

        href = link.get("href", "")
        title = link.get_text(strip=True)

        if not title:
            img = link.select_one("img")
            if img:
                title = img.get("alt", "").strip()
                
        title = html.unescape(title)

        if not href or not title:
            continue

        full_url = BASE_URL + href if href.startswith("/") else href

        if full_url in existing_urls:
            found_existing = True
            break

        if full_url in seen:
            continue

        seen.add(full_url)

        articles.append({
            "url": full_url,
            "title": title
        })

    return articles, found_existing


def get_article_urls(existing_urls: set, max_clicks: int = 10) -> list[dict]:
    print("[크롤러] AI타임스 목록 페이지 접속 중...")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.get(LIST_URL)
    time.sleep(2)

    all_articles = []
    seen = set()

    def extract_articles():
        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = []
        found_existing = False

        links = soup.select("ul.altlist-webzine a[href*='articleView']")

        for link in links:
            href = link.get("href", "")
            full_url = BASE_URL + href if href.startswith("/") else href

            if full_url in seen:
                continue

            if full_url in existing_urls:
                found_existing = True
                break

            title = link.get_text(strip=True)

            if not title:
                img = link.select_one("img")
                if img:
                    title = img.get("alt", "").strip()

            title = html.unescape(title)

            if not full_url or not title:
                continue

            seen.add(full_url)

            articles.append({
                "url": full_url,
                "title": title
            })

        return articles, found_existing

    first_articles, found_existing = extract_articles()
    all_articles.extend(first_articles)

    print(f"  → 초기 기사 {len(first_articles)}개 수집")

    if found_existing:
        print("  → 기존 기사 발견. 종료")
        driver.quit()
        return all_articles

    for i in range(max_clicks):
        try:
            before_count = len(
                driver.find_elements(
                    By.CSS_SELECTOR,
                    "ul.altlist-webzine a[href*='articleView']"
                )
            )

            more_button = driver.find_element(By.CSS_SELECTOR, "button.list-btn-more")

            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});",
                more_button
            )

            time.sleep(1)
            driver.execute_script("arguments[0].click();", more_button)

            WebDriverWait(driver, 10).until(
                lambda d: len(
                    d.find_elements(
                        By.CSS_SELECTOR,
                        "ul.altlist-webzine a[href*='articleView']"
                    )
                ) > before_count
            )

            new_articles, found_existing = extract_articles()
            all_articles.extend(new_articles)

            print(
                f"  → 더보기 {i + 1}회 클릭: "
                f"신규 {len(new_articles)}개 / 누적 {len(all_articles)}개"
            )

            if found_existing:
                print("  → 기존 기사 발견. 종료")
                break

        except Exception as e:
            print(f"  → 더보기 실패: {e}")
            break

    driver.quit()

    print(f"  → 총 {len(all_articles)}개 기사 URL 추출")
    return all_articles


def extract_published_date(soup: BeautifulSoup) -> str:
    """기사 입력 날짜 추출: 2026-05-06 13:58 형태로 반환"""
    published_date = "날짜 미상"

    # 1순위: 업데이트 기사에서 입력 날짜가 별도 div에 있는 경우
    date_tag = soup.select_one("div.info-update-origin")

    if date_tag:
        raw_text = date_tag.get_text(" ", strip=True)
    else:
        # 2순위: breadcrumbs 안의 li 중 "입력"이 들어간 li 찾기
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
            raw_text
        )

        if match:
            published_date = (
                match.group(0)
                .replace(".", "-")
                .replace("/", "-")
            )

    return published_date


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


def crawl(
    max_clicks: int = 10000,
    save_path: str = "./data/raw_articles.json"
) -> list[dict]:

    print("=" * 40)
    print("AI타임스 크롤링 시작")
    print("=" * 40)

    # 폴더 없으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # json 없으면 새 파일 생성
    if not os.path.exists(save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    # 기존 데이터 로드
    with open(save_path, "r", encoding="utf-8") as f:
        existing_articles = json.load(f)

    existing_urls = {
        article["url"]
        for article in existing_articles
    }

    # 신규 URL 수집
    article_list = get_article_urls(
        existing_urls=existing_urls,
        max_clicks=max_clicks
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
            article["title"]
        )

        if data:
            new_results.append(data)

        time.sleep(0.5)

    # 기존 + 신규 합치기
    all_articles = existing_articles + new_results

    # 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            all_articles,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\n✅ 기존 {len(existing_articles)}개 + "
        f"신규 {len(new_results)}개 = "
        f"총 {len(all_articles)}개 저장"
    )

    return new_results


if __name__ == "__main__":
    crawl(10)