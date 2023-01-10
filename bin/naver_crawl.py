# 크롤링시 필요한 라이브러리 불러오기
from bs4 import BeautifulSoup
import requests
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm


# 웹드라이버 설정
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)


# 페이지 url 형식에 맞게 바꾸어 주는 함수 만들기
# 입력된 수를 1, 11, 21, 31 ...만들어 주는 함수
def makePgNum(num):
    if num == 1:
        return num
    elif num == 0:
        return num + 1
    else:
        return num + 9 * (num - 1)


# 크롤링할 url 생성하는 함수 만들기(검색어, 크롤링 시작 페이지, 크롤링 종료 페이지)
def makeUrl(search, start_pg, end_pg):
    if start_pg == end_pg:
        start_page = makePgNum(start_pg)
        url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(
            start_page)
        print("생성url: ", url)
        return url
    else:
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = makePgNum(i)
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(page)
            urls.append(url)
        print("생성url: ", urls)
        return urls


##########뉴스크롤링 시작###################

# 검색어 입력
search = input("검색할 키워드를 입력해주세요:")

# 검색 시작할 페이지 입력
page = int(input("\n크롤링할 시작 페이지를 입력해주세요. ex)1(숫자만입력):"))  # ex)1 =1페이지,2=2페이지...
print("\n크롤링할 시작 페이지: ", page, "페이지")
# 검색 종료할 페이지 입력
page2 = int(input("\n크롤링할 종료 페이지를 입력해주세요. ex)1(숫자만입력):"))  # ex)1 =1페이지,2=2페이지...
print("\n크롤링할 종료 페이지: ", page2, "페이지")

# naver url 생성
search_urls = makeUrl(search, page, page2)

## selenium으로 navernews만 뽑아오기##

# 버전에 상관 없이 os에 설치된 크롬 브라우저 사용
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.implicitly_wait(1.5)

# selenium으로 검색 페이지 불러오기 #

naver_urls = []

for i in search_urls:
    driver.get(i)
    time.sleep(1)  # 대기시간 변경 가능

    # 네이버 기사 눌러서 제목 및 본문 가져오기#
    # 네이버 기사가 있는 기사 css selector 모아오기
    a = driver.find_elements(By.CSS_SELECTOR, 'a.info')

    # 위에서 생성한 css selector list 하나씩 클릭하여 본문 url얻기
    for i in a:
        i.click()

        # 현재탭에 접근
        driver.switch_to.window(driver.window_handles[1])
        time.sleep(3)  # 대기시간 변경 가능

        # 네이버 뉴스 url만 가져오기

        url = driver.current_url
        print(url)

        if "news.naver.com" in url:
            naver_urls.append(url)

        else:
            pass
        # 현재 탭 닫기
        driver.close()
        # 다시처음 탭으로 돌아가기(매우 중요!!!)
        driver.switch_to.window(driver.window_handles[0])

print(naver_urls)

###naver 기사 본문 및 제목 가져오기###

# ConnectionError방지
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

titles = []
dates=[]
contents = []
for i in naver_urls:
    original_html = requests.get(i, headers=headers)
    html = BeautifulSoup(original_html.text, "html.parser")
    # 검색결과확인시
    # print(html)

    # 뉴스 제목 가져오기
    title = html.select("div#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
    # list합치기
    title = ''.join(str(title))
    # html태그제거
    pattern1 = '<[^>]*>'
    title = re.sub(pattern=pattern1, repl='', string=title)
    titles.append(title)

    date = html.select("span.media_end_head_info_datestamp_time_.ARTICLE_DATE_TIME")
    date =''.join(str(title))
    pattern1 = '<[^>]*>'
    date = re.sub(pattern=pattern1, repl='', string=date)
    dates.append(date)
    print(date)
    

    # 뉴스 본문 가져오기

    content = html.select("div#dic_area")

    # 기사 텍스트만 가져오기
    # list합치기
    content = ''.join(str(content))

    # html태그제거 및 텍스트 다듬기
    content = re.sub(pattern=pattern1, repl='', string=content)
    pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
    content = content.replace(pattern2, '')

    contents.append(content)
    print(title)
    print(date)
    print(content[:10])

print(titles)
print(dates)
print(contents)

# 데이터프레임으로 정리(titles,url,contents)
import pandas as pd

news_df = pd.DataFrame({'title': titles, 'link': naver_urls, 'content': contents})

news_df.to_csv('NaverNews_%s.csv' % search, index=False, encoding='utf-8-sig')