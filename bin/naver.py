# 크롤링시 필요한 라이브러리 불러오기
from bs4 import BeautifulSoup
import requests
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
from tqdm import tqdm
import pandas as pd


class Naver:
    def __init__(self):
        self.base_url="https://search.naver.com/search.naver?where=news&sm=tab_pge&query={0}&sort=0&photo=0&field=0&pd=3&ds={1}&de={2}&cluster_rank=72&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from{3}to{4},a:all&start={5}"
        self.date_ranges = []
        self.search_urls=[]
        self.naver_urls=[]
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

    def run(self):
        self.set_dates('2020.01.01')
        #first, make all urls to access
        for date in self.date_ranges:

            self.naver_urls = []

            urls = self.makeurl('반도체', 10, 11, date[0], date[1])
            # self.search_urls = self.search_urls + urls
        
            #from the urls, get naver news urls
            for search_url in urls:
                n_url = self.find_naver(search_url)
                self.naver_urls = self.naver_urls + n_url

             #from each self.naver url, scrape title, article
            titles, dates, contents = self.collect_art(self.naver_urls)
            
            news_df = pd.DataFrame({'title': titles, 'dates': dates, 'content': contents})
            news_df.to_csv('News_'+str(date[0]) + '_'+str(date[1])+'.csv', index=False, encoding='utf-8-sig')


    def makePgNum(self, num):
        if num == 1:
            return num
        elif num == 0:
            return num + 1
        else:
            return num + 9 * (num - 1)

    def makeurl(self, keyword, start_pg, end_pg, start_date, end_date):
        urls=[]
        sec_start = start_date.replace('.','')
        sec_end = end_date.replace('.','')
        for i in range(start_pg, end_pg+1):
            page= self.makePgNum(i)
            url = self.base_url.format(keyword,start_date, end_date, sec_start, sec_end, page)
            urls.append(url)
        
        return urls

    def set_dates(self, start_date):
        dates=[]
        dates.append(['2021.01.01','2021.12.31'])
        self.date_ranges = dates
        return dates

    def find_naver(self, search_url):

        options_ = webdriver.ChromeOptions()
        options_.add_experimental_option("excludeSwitches", ["enable-logging"])
        options_.add_argument('headless')
        
        driver = webdriver.Chrome('./chromedriver',options=options_)
        driver.implicitly_wait(3)
        
        naver_urls=[]

        driver.get(search_url)
        a = driver.find_elements(By.CSS_SELECTOR, 'a.info') 
        for i in a:
            i.click()

            # 현재탭에 접근
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(2)  # 대기시간 변경 가능

            # 네이버 뉴스 url만 가져오기

            url = driver.current_url
            # print(url)

            if "news.naver.com" in url:
                naver_urls.append(url)
                print(url)

            else:
                pass
            # 현재 탭 닫기
            driver.close()
            # 다시처음 탭으로 돌아가기(매우 중요!!!)
            driver.switch_to.window(driver.window_handles[0])

        return naver_urls

    def collect_art(self, urls):
        titles=[]
        contents=[]
        dates=[]

        for i in urls:
            print(i)
            original_html = requests.get(i, headers=self.headers)
            html = BeautifulSoup(original_html.text, "html.parser")


            # 뉴스 제목 가져오기
            title = html.select("div#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
            # list합치기
            title = ''.join(str(title))
            # html태그제거
            pattern1 = '<[^>]*>'
            title = re.sub(pattern=pattern1, repl='', string=title)
            titles.append(title)

            date = html.select('span#media_end_head_info_datestamp_time._ARTICLE_DATE_TIME')
            title = ''.join(str(title))
            pattern1 = '<[^>]*>'
            date = re.sub(pattern=pattern1, repl='', string=date)
            dates.apend(date)
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
        print(dates)
        # print(content[:10])
        return titles, dates, contents


if __name__=='__main__':
    N = Naver()
    N.run()