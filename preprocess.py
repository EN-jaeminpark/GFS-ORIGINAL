import requests
import os
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import yfinance as yf
from pytrends.request import TrendReq
import time
from utils.get_price import *
from utils.get_trends import *
from utils.match_price_and_trend import *
from utils.get_keywords import *
from utils.make_features import *

#################################### ARGPARSE ###############################################
parser = argparse.ArgumentParser(description='Prepare Data')
parser.add_argument('--stocks', default='./stocks_info.csv', help='stocks list csv file path')
parser.add_argument('--start_date', default='2019-01-01', help='start date')
parser.add_argument('--mid_date', default='2019-07-01', help='mid date (6m time diff)')
parser.add_argument('--end_date', default='2019-12-31', help='end date')
parser.add_argument('--news_topic', default='nasdaq', type=str, help='news search keyword')
parser.add_argument('--n', default='1', type=str, help='dataset number')
args = parser.parse_args()
#############################################################################################

if __name__ == '__main__':
    stocks_df = pd.read_csv(args.stocks)
    
    # first, collect price data using Yahoo Finance
    get_price(args.n, stocks_df, args.start_date, args.end_date)

    # Second, collect trend data for each ticker using Google Trends (Pytrends)
    get_trends(args.n, stocks_df, args.start_date, args.mid_date, args.end_date)

    # Third, extract 5 keywords from designated news topic and collect trends
    get_keywords(args.n, args.news_topic, args.start_date, args.mid_date, args.end_date)

    # Fourth, match stock dates and trends dates (stock tickers and keywords)
    match_price_and_trend(args.n)

    # Fifth, make price features and trend features
    make_price_features(args.n)
    make_trend_features(args.n)

    # Sixth, make merged versions of ticker trends
    make_merged_trends(args.n)








