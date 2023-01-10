import requests
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from pytrends.request import TrendReq
import time
from utils.get_data import *

#################################### ARGPARSE #######################################
parser = argparse.ArgumentParser(description='Prepare DATA')
parser.add_argument('--stocks', default='./stocks_info.csv', help='stocks list csv file path')
parser.add_argument('--start_date', default='2019-01-01', help='start date')
parser.add_argument('--mid_date', default='2019-07-01', help='mid date (6m time diff)')
parser.add_argument('--end_date', default='2019-12-31', help='end date')
args = parser.parse_args()
#####################################################################################

if __name__ == '__main__':
    stocks_df = pd.read_csv(args.stocks)

   


def make_price_features():
    price_folder = './Dataset/date_matchd/Price/'
    target_folder = './Dataset/Final/Price/'
    
    price_files = os.listdir(price_folder)
    scaler = MinMaxScaler()
    
    for file in price_files:
        print(file)
        price_df = pd.read_csv(price_folder + file, index_col=0)
    
        new_price_df = pd.DataFrame()

        c_open=[]
        c_high =[]
        n_close = []
        c_low= []
        
        five_day=[] 
        ten_day= []
        fifteen_day=[]
        twenty_day = []
        twentyfive_day =[]
        thirty_day =[]
        
        label = []
        
        for i in range(len(price_df)):
            if i<30:
                continue
            
            c_open.append(price_df['Open'][i]/price_df['Close'][i] - 1)
            c_high.append(price_df['High'][i]/price_df['Close'][i] - 1)
            c_low.append(price_df['Low'][i]/price_df['Close'][i] - 1)
            n_close.append(price_df['Close'][i]/price_df['Close'][i-1] - 1)
            
            five_day.append(price_df['Close'][i-5:i].sum()/price_df['Close'][i] - 1)         
            ten_day.append(price_df['Close'][i-10:i].sum()/price_df['Close'][i] - 1)
            fifteen_day.append(price_df['Close'][i-15:i].sum()/price_df['Close'][i] - 1)
            twenty_day.append(price_df['Close'][i-20:i].sum()/price_df['Close'][i] - 1)
            twentyfive_day.append(price_df['Close'][i-25:i].sum()/price_df['Close'][i] - 1)
            thirty_day.append(price_df['Close'][i-30:i].sum()/price_df['Close'][i] - 1)

            if (price_df['Close'][i]/price_df['Close'][i-1] - 1) > 0.001:
                label.append(1)
            else:
                label.append(0)

        new_price_df['c_open'] = c_open
        new_price_df['c_high'] = c_high
        new_price_df['c_low'] = c_low
        new_price_df['n_close'] = n_close

        new_price_df['five'] = five_day
        new_price_df['ten'] = ten_day
        new_price_df['fifteen'] = fifteen_day
        new_price_df['twenty'] = twenty_day
        new_price_df['twentyfive'] = twentyfive_day
        new_price_df['thirty'] = thirty_day

        new_price_df = scaler.fit_transform(new_price_df)
        new_price_df = pd.DataFrame(new_price_df)
        
        new_price_df['label'] = label

        new_price_df.to_csv(target_folder + file, index=False, header=None)
        print(new_price_df)
        
def make_trend_features():
    #FOR 5 GLOBAL KEYWORDS ONLY
    folder_path ='./Dataset/date_matchd/keywords/'
    files = os.listdir(folder_path)
    
    for file in files:
        
        print(file)
        tmp_df = pd.read_csv(folder_path + file, index_col=0)
        name = file.split('.')[0]
        tmp_df = tmp_df.reset_index()
        new_price_df = pd.DataFrame()
        price_df =tmp_df
        
        

           
           
        one_day=[] 
        five_day = []
        ten_day= []
        fifteen_day=[]
        twenty_day = []
        twentyfive_day =[]
        thirty_day =[]
        
        # label = []
        # print(price_df)
        # exit()
        for i in range(len(price_df)):
            
            if i<30:
                continue
            
            one_day.append(price_df[name][i])
            
            five_day.append(price_df[name][i-5:i].sum()/5)         
            ten_day.append(price_df[name][i-10:i].sum()/10)
            fifteen_day.append(price_df[name][i-15:i].sum()/15)
            twenty_day.append(price_df[name][i-20:i].sum()/20)
            twentyfive_day.append(price_df[name][i-25:i].sum()/25)
            thirty_day.append(price_df[name][i-30:i].sum()/30)

            # if (price_df['Close'][i]/price_df['Close'][i-1] - 1) > 0.001:
            #     label.append(1)
            # else:
            #     label.append(0)
            
        new_price_df['one'] = one_day
        new_price_df['five'] = five_day
        new_price_df['ten'] = ten_day
        new_price_df['fifteen'] = fifteen_day
        new_price_df['twenty'] = twenty_day
        new_price_df['twentyfive'] = twentyfive_day
        new_price_df['thirty'] = thirty_day

        # new_price_df = scaler.fit_transform(new_price_df)
        new_price_df = pd.DataFrame(new_price_df)
        
        # new_price_df['label'] = label

        new_price_df.to_csv('./Dataset/Final/Keywords_trend/' + file, index=False, header=None)
        print(new_price_df)
        
def redefine_stock_trends():
    folder_path = './Dataset/Final/Stocks_trend/'
    files = os.listdir(folder_path)
    for file in files:
        df = pd.read_csv(folder_path + file)
        new_df = pd.DataFrame()
        name = file.split('.')[0]

        new_df = df[name]

        new_df.to_csv(folder_path+file, header=None, index=False)

def make_merged_trends():
    files = os.listdir('./Dataset/Final/Stocks_trend/')
    companies = os.listdir('./Dataset/Final/Price/')
    new_trends_df = pd.DataFrame()
    for i, trend in enumerate(files):
            
        if trend not in companies:
            continue

        comp_name = trend.split('.')[0]
        trends_df = pd.read_csv('./Dataset/Final/Stocks_trend/' + trend, header=None)
        trends = trends_df[30:]
        print(trends)
        
        # trends = trends_df["Change"]

        
        new_trends_df[comp_name] = trends

    new_trends_df.to_csv('./Dataset/Final/Stocks_merged.csv', index=False, header=None)

if __name__=='__main__':
    get_price()        
    get_trend()
    match_price_and_trend()

    make_price_features()
    make_trend_features()
    redefine_stock_trends()
    make_merged_trends()
    make_merged_keywords_trends()
    
