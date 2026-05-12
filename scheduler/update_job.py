import subprocess
import sys

def weekly_update():
    print("크롤링 시작")
    subprocess.run([sys.executable, "-m", "crawler.crawl"])
    print("크롤링 완료")
    
    print("임베딩 시작")
    subprocess.run([sys.executable, "-m", "rag.embedder"])
    print("임베딩 완료")
    
    print("git 추가")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "weekly ai news update"], check=False)
    subprocess.run(["git", "push"], check=True)
    print("git 추가 완료")

if __name__ == "__main__":
    weekly_update()