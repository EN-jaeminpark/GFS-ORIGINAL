import pandas as pd
import numpy as np
import requests
import os
import pickle
import re
from pytrends.request import TrendReq
import time
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

def get_keywords(data_num, news_topic, start_date, mid_date, end_date):
    keywords_folder = './Dataset_' + str(data_num) + '/Keywords/'
    if not os.path.exists(keywords_folder):
        os.makedirs(keywords_folder)
    
    # read total news file and sort those only fall under the selected time period
    # news_df = pd.read_csv('./News/' + str(news_topic) + '.csv')
    news_df = pd.read_csv('./News/' + str(news_topic) + '.csv')
    year = start_date.split('-')[0]
    articles=[]
    
    for i in range(len(news_df)):
        # print(news_df['Date'][i])
        if year in news_df['Date'][i]:
            articles.append(news_df['Article'][i])

    print('Number of articles: ', len(articles))
    # Calculate TF-IDF score to get top 5 keywords
    f = open('./reference/news_stopwords.txt','r')
    news_stopwords = f.readline()
    f.close()

    stop_words = set(stopwords.words("english"))
    
    # filter words
    # add tokens to create total tokens list and filter word for each article
    ppd_list=[]
    total_tokens=[]
    for article in articles:
        filtered_words=[]
        ppd_txt = re.sub('[^a-zA-Z]', ' ', str(article))
        word_tokens = word_tokenize(ppd_txt)
        lower_words = [w.lower() for w in word_tokens]
        
        for word in lower_words:
            if word not in stop_words and word not in news_stopwords and len(word) > 1 and word != 'nan':
                filtered_words.append(word)
                total_tokens.append(word)
        
        filtered_dict = dict(Counter(filtered_words))
        ppd_list.append(filtered_dict)
    
    total_tokens = list(set(total_tokens))
    print('Collected tokens num: ', len(total_tokens))
    print('Now extracting top 5 TF-IDF keywords from articles...')
    
    doc_num = len(ppd_list)
    top5=[]
    for token in tqdm(total_tokens):
        TF_SCORE = 0
        IDF_SCORE = 0
        doc_cnt = 0

        for article in ppd_list:
            if token in article:
                TF_SCORE += article[token] / sum(article.values())
                doc_cnt+=1
        
        IDF_SCORE = doc_num / (doc_cnt + 1)

        # TFIDF_SCORE = np.log(TF_SCORE) + np.log(IDF_SCORE)
        TFIDF_SCORE = TF_SCORE * IDF_SCORE
        score_set = [token, TFIDF_SCORE]

        if len(top5) < 5:
            top5.append(score_set)
        else:
            sorted_list = sorted(top5, key=lambda x: x[1], reverse=True)
            if TFIDF_SCORE > sorted_list[-1][1]:
                top5.remove(sorted_list[-1])
                top5.append(score_set)

    print('--------------------- TOP 5 KEYWORDS ----------------------')
    print(top5)
    
    if str(data_num) == '99':
        top5 = ['economy', 'nasdaq', 'currency', 'debt', 'stock']

    # Collect Google Trends for each of the selected keywords
    pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.1)
    timeranges = [start_date+' '+ mid_date, mid_date + ' '+ end_date]
    for key in tqdm(top5):
        keyword = key[0]
        if os.path.exists(keywords_folder + keyword +'.csv'):
            print(keywords_folder + keyword +'.csv already exists')
            continue
        
        kw_list = [keyword]
        print('Working on.... ', keyword)

        for a, timerange in enumerate(timeranges):
            try:
                time.sleep(1)
                pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
            except requests.exceptions.Timeout:
                    time.sleep(30)
                    pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
            except pytrends.exceptions.ResponseError:
                time.sleep(30)
                pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
        
            interest_over_time = pytrends.interest_over_time()
            interest_over_time = interest_over_time.reset_index()

            if a==0:
                df = interest_over_time
            else:
                trend_total = pd.DataFrame()
                
                try:
                    first_num = list(interest_over_time[keyword])[0]
                except:
                    pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
                    interest_over_time = pytrends.interest_over_time()
                    interest_over_time = interest_over_time.reset_index()
                    first_num = list(interest_over_time[keyword])[0]
                try:
                    last_num = list(df[keyword])[-1]
                except:
                    pytrends.build_payload(kw_list, cat=0, timeframe=timeranges[0])
                    df = pytrends.interest_over_time()
                    df = interest_over_time.reset_index()
                    last_num = list(df[keyword])[-1]
                
                ratio = float(last_num / (first_num+1))
                if ratio == 0:
                    ratio = 1
                interest_over_time[keyword] = interest_over_time[keyword].multiply(ratio)
                final = pd.concat([df, interest_over_time.iloc[1:,:]], axis=0)

                changes=[]
                for i in range(len(final)):
                    if i ==0:
                        changes.append(0)
                    else:
                        ft = list(final[keyword])
                        change = (ft[i]-ft[i-1]) / (ft[i-1]+1)
                        changes.append(change)
                
                trend_total['date'] = final["date"]
                trend_total[keyword] = changes

                trend_total.to_csv(keywords_folder + keyword +'.csv')
                print(trend_total)

if __name__ == '__main__':
    get_keywords(1, 'nasdaq', '2020-01-01', '2020-07-01', '2020-12-31')