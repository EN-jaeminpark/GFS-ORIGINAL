import time
import requests
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

def make_price_features(data_num):
    price_folder ='./Dataset_' + str(data_num) + '/Price/'
    final_folder = './Dataset_' + str(data_num) + '/Final/Price/'
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    price_files = os.listdir(price_folder)
    print('Creating price features...')
    for file in tqdm(price_files):
        file_path = price_folder + file
        price_df = pd.read_csv(file_path, index_col=0)

        new_price_df = pd.DataFrame()

        c_open=[]
        c_high =[]
        c_close = []
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
            c_close.append(price_df['Close'][i]/price_df['Close'][i-1] - 1)
            
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
        new_price_df['c_close'] = c_close

        new_price_df['five'] = five_day
        new_price_df['ten'] = ten_day
        new_price_df['fifteen'] = fifteen_day
        new_price_df['twenty'] = twenty_day
        new_price_df['twentyfive'] = twentyfive_day
        new_price_df['thirty'] = thirty_day
        
        new_price_df['label'] = label

        new_price_df.to_csv(final_folder+file, index=False, header=None)

def make_trend_features(data_num):
    folder_path = './Dataset_' + str(data_num) + '/Keywords/'
    final_folder = './Dataset_' + str(data_num) + '/Final/Keywords/'
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    files = os.listdir(folder_path)
    print('Creating keyword trend features...')
    for file in tqdm(files):
        trend_df = pd.read_csv(folder_path + file, index_col=0)
        name = file.split('.')[0]
        trend_df =trend_df.reset_index()

        new_df = pd.DataFrame()

        one_day=[] 
        five_day = []
        ten_day= []
        fifteen_day=[]
        twenty_day = []
        twentyfive_day =[]
        thirty_day =[]

        
        for i in range(len(trend_df)):
            
            if i<30:
                continue
            
            one_day.append(trend_df[name][i])
            
            five_day.append(trend_df[name][i-5:i].sum()/5)         
            ten_day.append(trend_df[name][i-10:i].sum()/10)
            fifteen_day.append(trend_df[name][i-15:i].sum()/15)
            twenty_day.append(trend_df[name][i-20:i].sum()/20)
            twentyfive_day.append(trend_df[name][i-25:i].sum()/25)
            thirty_day.append(trend_df[name][i-30:i].sum()/30)
            
        new_df['one'] = one_day
        new_df['five'] = five_day
        new_df['ten'] = ten_day
        new_df['fifteen'] = fifteen_day
        new_df['twenty'] = twenty_day
        new_df['twentyfive'] = twentyfive_day
        new_df['thirty'] = thirty_day

        new_df = pd.DataFrame(new_df)
        
        new_df.to_csv(final_folder + file, index=False, header=None)

def make_merged_trends(data_num):
    # make merged csv file of each ticker trend file
    # this makes it easier to calculate corr in the graph construction step
    # however, do not include ticker if not in Price folder

    price_files = os.listdir('./Dataset_'+str(data_num)+'/Price/')
    
    trend_folder = './Dataset_'+str(data_num)+'/Trend/'
    final_folder = './Dataset_'+str(data_num)+'/Final/Trend/'
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    trend_files = os.listdir(trend_folder)

    new_df = pd.DataFrame()
    print('Merging ticker trends...')
    for i, trend in enumerate(tqdm(trend_files)):
        
        if trend not in price_files:
            continue
        
        comp_name = trend.split('.')[0] 
        trend_df = pd.read_csv(trend_folder + trend)
        
        trend = trend_df[comp_name][30:]
        
        new_df[comp_name] = trend
    
    new_df.to_csv(final_folder+'stocks_merged.csv', index=False, header=None)

        
