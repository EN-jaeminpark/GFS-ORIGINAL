import requests
import os
import pickle
import pandas as pd
import numpy as np
import time
from pytrends.request import TrendReq

def get_trends(data_num, stocks_df, start_date, mid_date, end_date):
    trend_folder = './Dataset_'+ data_num + '/Trend/' 
    if not os.path.exists(trend_folder):
        os.makedirs(trend_folder)
    price_folder = './Dataset_'+ data_num + '/Price/'
    pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.1)
    price_list =os.listdir(price_folder)

    for idx in range(len(stocks_df)):
        if os.path.exists(trend_folder + stocks_df['Ticker'][idx]+'.csv'):
            print('GTrends for {0} already exists'.format(stocks_df['Ticker'][idx]))
            continue

        tic = stocks_df['Ticker'][idx]
        if tic+'.csv' in price_list:
            pass
        else:
            continue
        timeranges = [start_date+' '+ mid_date, mid_date + ' '+ end_date]

        for a, timerange in enumerate(timeranges):
            kw_list = [stocks_df['Ticker'][idx]]
            
            try:
                time.sleep(10)
                pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
            except requests.exceptions.Timeout:
                    time.sleep(60)
                    pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
            except pytrends.exceptions.ResponseError:
                time.sleep(60)
                pytrends.build_payload(kw_list, cat=0, timeframe=timerange)

            interest_over_time = pytrends.interest_over_time()
            interest_over_time = interest_over_time.reset_index()

            if a==0:
                df = interest_over_time
            else:
                trend_total = pd.DataFrame()

                tic = stocks_df['Ticker'][idx]
                
                first_num = list(interest_over_time[tic])[0]
                last_num = list(df[tic])[-1]
                
                ratio = float(last_num / (first_num+1))
                interest_over_time[tic] = interest_over_time[tic].multiply(ratio)
                final = pd.concat([df, interest_over_time.iloc[1:,:]], axis=0)

                changes=[]
                for i in range(len(final)):
                    if i ==0:
                        changes.append(0)
                    else:
                        ft = list(final[tic])
                        change = (ft[i]-ft[i-1]) / (ft[i-1]+1)
                        changes.append(change)
                
                trend_total['date'] = final["date"]
                trend_total[tic] = changes

                trend_total.to_csv(trend_folder + tic +'.csv')
                
                if idx == 0:
                    print('-- Trend csv example --')
                    print(trend_total)

                time.sleep(5)
