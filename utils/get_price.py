import pandas as pd
import numpy as np
from tqdm import tqdm
import yfinance as yf
import os

def get_price(data_num, stocks_df, start_date, end_date):

    if not os.path.exists('./Dataset_' + str(data_num)):
        os.makedirs('./Dataset_' + str(data_num))

    
    price_folder = './Dataset_'+ str(data_num) + '/Price/'
    if not os.path.exists(price_folder):
        os.makedirs(price_folder)
    fail_list=[]

    for idx in range(len(stocks_df)):
        if os.path.exists(price_folder + stocks_df['Ticker'][idx]+'.csv'):
            continue
        
        data = yf.Ticker(stocks_df['Ticker'][idx])
        
        try:
            data = data.history(start = start_date, end = end_date)
            print(data)
        except:
            fail_list.append(stocks_df['Ticker'][idx])
            continue

        data = data.iloc[:,:4]

        if len(data) > 200:
            data.to_csv(price_folder + stocks_df['Ticker'][idx]+'.csv')
        else:
            fail_list.append(stocks_df['Ticker'][idx])


    print('------ Finished collecting price data ------')
    print('Failed collecting price for: ', fail_list)
