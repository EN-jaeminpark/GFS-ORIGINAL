import pandas as pd
import numpy as np
import os

def match_price_and_trend(data_num):
    base_folder = './Dataset_' + str(data_num)
    price_folder = base_folder + '/Price/'
    trends_folder = base_folder +'/Trend/'
    keywords_folder =base_folder +'/Keywords/'

    # Open a price file to get trading days
    price_files = os.listdir(price_folder)
    price_basis = pd.read_csv(price_folder + price_files[0])
    trading_days = price_basis['Date']
    days = [d.split(' ')[0] for d in trading_days]

    # For each trends(ticker) file, eliminate non-trading days
    trend_files = os.listdir(trends_folder)
    for file in trend_files:
        file_path = trends_folder + file
        trend_df = pd.read_csv(file_path)
        dates = list(trend_df['date'])
        idx_list =[]
        for i, d in enumerate(dates):
            if d not in days:
                idx_list.append(i)
        trend_df = trend_df.drop(trend_df.index[idx_list])
        trend_df.to_csv(file_path)


    # for each trends(keyword) file, eliminate non-trading days
    keyword_files = os.listdir(keywords_folder)
    for file in keyword_files:
        file_path = keywords_folder + file
        trend_df = pd.read_csv(file_path)
        dates = list(trend_df['date'])
        idx_list =[]
        for i, d in enumerate(dates):
            if d not in days:
                idx_list.append(i)
        trend_df = trend_df.drop(trend_df.index[idx_list])
        trend_df.to_csv(file_path)

