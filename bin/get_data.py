import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import pickle
from pytrends.request import TrendReq
import time



# print(semicon_df)
stocks_df = pd.read_csv('./stocks_info.csv')
def get_price():
    price_folder = './Dataset/Price/'
    for idx in range(len(semicon_df)):
        ticker = semicon_df['Ticker'][idx]
        
        data = yf.Ticker(ticker)
        data = data.history(start="2021-01-01",  end="2021-12-31")
        
        if len(data) != 251:
            # print(ticker)
            # print(data)
            continue
        else:
            data = data.iloc[:,:4]
            data.to_csv(price_folder + ticker+'.csv')

def get_trend():
    # trend_folder = './Dataset/Trend/'
    trend_folder = './Dataset/Raw/keywords/'
    pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.1)
    
    # key_list = ['semiconductor stock', 'TSMC', 'diode', 'SOX', 'semiconductor shortage']
    
    for idx in range(len(semicon_df)):
        ticker = semicon_df['Ticker'][idx]
    # for ticker in key_list:    
        if ticker == 'MU' or ticker=='GFS':
            continue
        else:
            
            timeranges=['2021-01-01 2021-07-01','2021-07-01 2021-12-31']
            df = pd.DataFrame()
            for a, timerange in enumerate(timeranges):
                
                kw_list = [ticker]
                if os.path.exists(trend_folder+ticker+'.csv'):
                    print('file exists: ', ticker)
                    continue
                
                # time.sleep(10)
                try:
                    time.sleep(1)
                    pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
                except requests.exceptions.Timeout:
                    time.sleep(30)
                    try:
                        pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
                    except:
                        print('Error occured with pytrend')
                        exit()
                except pytrends.exceptions.ResponseError:
                    time.sleep(30)
                    pytrends.build_payload(kw_list, cat=0, timeframe=timerange)
                
                interest_over_time = pytrends.interest_over_time()
                interest_over_time = interest_over_time.reset_index()
                # interest_over_time = interest_over_time.drop('isPartial',axis=1)
                # interest_over_time = interest_over_time.reset_index()
                
                if a == 0:
                    df = interest_over_time
                else:
                    real_final=pd.DataFrame()
                    last_num = list(df[ticker])[-1]
                    first_num = list(interest_over_time[ticker])[0]
                    
                    ratio = float(last_num / (first_num+1))
                    interest_over_time[ticker] = interest_over_time[ticker].multiply(ratio)
                    
                    final = pd.concat([df, interest_over_time.iloc[1:, :]], axis=0)

                    changes=[]
                    for i in range(len(final)):
                        if i==0:
                            changes.append(0)
                        else:
                            ft = list(final[ticker])
                            
                            # final=pd.DataFrame(final)
                            change = (ft[i] - ft[i-1]) / (ft[i-1] + 1)
                            # change = (final[ticker][i] - final[ticker][i-1]) / (final[ticker][i-1] + 1)
                            changes.append(change)
                    
                    real_final['date'] = final['date']
                    real_final[ticker] = changes

                    real_final.to_csv(trend_folder + ticker +'.csv') 
                    print(real_final) 
                    print(ticker)              
                    time.sleep(10)
            
            
                
def match_price_and_trend():
    ex_df = pd.read_csv('./Dataset/date_matchd/Price/AAPL.csv')
    trading_days = ex_df['Date']                
    trend_files = os.listdir('./Dataset/Raw/keywords/')
    days=[]
    for day in trading_days:
        day = day.split(' ')[0]
        # print(day)
        days.append(day)
    
    for file in trend_files:
        # file_path = './Dataset/Trend/' + file
        file_path = './Dataset/Raw/keywords/' + file
        trend_df = pd.read_csv(file_path)
        dates = list(trend_df['date'])
        i_list = []
        for i, d in enumerate(dates):
            if d not in days:
                i_list.append(i)
                # trend_df=trend_df.drop(trend_df.index[i])
                print(d)
        trend_df = trend_df.drop(trend_df.index[i_list])
        trend_df.to_csv(file_path)
            
        
           
        
        
if __name__=='__main__':
    # get_price()        
    get_trend()
    match_price_and_trend()

