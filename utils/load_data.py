import numpy as np
import os
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yfinance as yf
import pickle
import warnings
warnings.simplefilter("ignore", UserWarning)
random_seed = 77
np.random.seed(random_seed)
torch.manual_seed(random_seed)


class Dataset_Loader:
    def __init__(self, price_folder, stock_trend_path, keyword_folder):
       self.price_folder = price_folder
       self.stock_trend_path = stock_trend_path
       self.keyword_folder = keyword_folder

       self.stocks_list = os.listdir(price_folder)
       self.keywords_list= os.listdir(keyword_folder)
       self.num_stocks = len(self.stocks_list)

    def run(self, seq, batch_size, train_start, valid_start, test_start, end):
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            self.data_separate(seq, train_start, valid_start, test_start, end)

        train_x, train_y, valid_x, valid_y, test_x, test_y      

        trainloader, validloader, testloader = self.create_data(batch_size,
        train_x[0], train_x[1], train_x[2], train_y, 
        valid_x[0], valid_x[1],valid_x[2], valid_y, 
        test_x[0],test_x[1], test_x[2], test_y)

        return trainloader, validloader, testloader

    def data_separate(self, seq, train_start, valid_start, test_start, end):

        total_price_data =[]        
        for stock in self.stocks_list:
            single_data = np.genfromtxt(
            os.path.join(self.price_folder, stock), dtype=float, delimiter=',',
            skip_header=False)
            total_price_data.append(single_data)

        total_keyword_data=[]
        for key in self.keywords_list:
            single_data = np.genfromtxt(
            os.path.join(self.keyword_folder, key), dtype=float, delimiter=',',
            skip_header=False)
            total_keyword_data.append(single_data)

        # total_stock_trend_data=[]
        # for key in os.listdir('./Dataset/Final/Stocks_trend/'):
        #     single_data = np.genfromtxt(
        #     os.path.join('./Dataset/Final/Stocks_trend', key), dtype=float, delimiter=',',
        #     skip_header=False)
        #     total_stock_trend_data.append(single_data)

        # print(len(total_price_data[0][0]))
        # print(len(total_price_data[0]))
        # print(len(total_price_data[2]))
        print(total_price_data[0][:])
        print(len(total_price_data[0][:, :5]))
        print('========================================================')    
        print('Total Price Data shape: ', torch.tensor(total_price_data).shape)  
        print('========================================================')

        
        print('========================================================')    
        print('Total Keyword Data shape: ', torch.tensor(total_keyword_data).shape)  
        print('========================================================')

        start_idx = seq
        train_fin_idx = int(len(total_price_data[0]) * 0.6)
        val_fin_idx = int(len(total_price_data[0]) * 0.8)
        end = len(total_price_data[0])
        
        train_num = (train_fin_idx - seq) * len(self.stocks_list)
        valid_num = (val_fin_idx - train_fin_idx) * len(self.stocks_list)
        test_num = (len(total_price_data[0]) - val_fin_idx) * len(self.stocks_list)
        price_feature_dim = total_price_data[0].shape[1] - 1 #-1 b/c of label
        keyword_feature_dim = total_keyword_data[0].shape[1]

        ############# 1. PRICE DATA ##################################################

        #make empty array
        price_train = np.zeros([train_num, seq, price_feature_dim], dtype=float)
        price_target = np.zeros([train_num, 1], dtype=float)

        ########### Train Data ##################
        ins_ind=0
        for idx in range(start_idx, train_fin_idx):
            for tic_idx in range(len(self.stocks_list)):
                price_train[ins_ind] = total_price_data[tic_idx][idx-seq: idx, : -1]
                price_target[ins_ind, 0] = (total_price_data[tic_idx][idx][-1]) 

                ins_ind+=1
                
        print('number of 0s in train: ', list(price_target).count(0))
        print('number of 1s in train: ',list(price_target).count(1))

        
        ########## Valid Data ####################
        ins_ind=0
        price_valid = np.zeros([valid_num, seq, price_feature_dim], dtype=float)
        target_valid = np.zeros([valid_num, 1], dtype=float)
        
        # for idx in range(train_end, valid_end):
        for idx in range(train_fin_idx, val_fin_idx):
            for tic_idx in range(len(self.stocks_list)):
                price_valid[ins_ind] = total_price_data[tic_idx][idx-seq: idx, : -1]
                # target_valid[ins_ind, 0] = (total_data[tic_idx][idx][-1] + 1)
                target_valid[ins_ind, 0] = (total_price_data[tic_idx][idx][-1])
                ins_ind+=1
                    
        print('number of 0s in valid: ', list(target_valid).count(0))
        print('number of 1s in valid: ', list(target_valid).count(1))
        

        ############ TEST Data #####################
        ins_ind=0
        price_test = np.zeros([test_num, seq, price_feature_dim], dtype=float)
        target_test = np.zeros([test_num, 1], dtype=float)        
        
        for idx in range(val_fin_idx, end):
            # for idx in range(valid_end, test_end):
            for tic_idx in range(len(self.stocks_list)):
                price_test[ins_ind] = total_price_data[tic_idx][idx - seq: idx, : -1]
                # target_test[ins_ind, 0] = (total_data[tic_idx][idx][-1] + 1)
                target_test[ins_ind, 0] = (total_price_data[tic_idx][idx][-1])
                ins_ind+=1

        print('number of 0s in test: ', list(target_test).count(0))
        print('number of 1s in test: ', list(target_test).count(1))

        ############################################################################
        ########## 2. KEYWORD TREND DATA ############################################
        ############################################################################
        train_num = (train_fin_idx - seq) * len(self.keywords_list)
        valid_num = (val_fin_idx - train_fin_idx) * len(self.keywords_list)
        test_num = (len(total_keyword_data[0]) - val_fin_idx) * len(self.keywords_list)

        keyword_train = np.zeros([train_num, seq, keyword_feature_dim], dtype=float)
        ########### Train Data ##################
        ins_ind=0
        for idx in range(start_idx, train_fin_idx):
            for tic_idx in range(len(self.keywords_list)):
                keyword_train[ins_ind] = total_keyword_data[tic_idx][idx-seq: idx, : ]
        
                ins_ind+=1
               
        ########## Valid Data ####################
        ins_ind=0
        keyword_valid = np.zeros([valid_num, seq, keyword_feature_dim], dtype=float)
                
        # for idx in range(train_end, valid_end):
        for idx in range(train_fin_idx, val_fin_idx):
            for tic_idx in range(len(self.keywords_list)):
                keyword_valid[ins_ind] = total_keyword_data[tic_idx][idx-seq: idx, :]
                
                ins_ind+=1

        ############ TEST Data #####################
        ins_ind=0
        keyword_test = np.zeros([test_num, seq, keyword_feature_dim], dtype=float)   
        
        for idx in range(val_fin_idx, end):
            # for idx in range(valid_end, test_end):
            for tic_idx in range(len(self.keywords_list)):
                keyword_test[ins_ind] = total_keyword_data[tic_idx][idx - seq: idx, : ]
                
                ins_ind+=1

        ###########################################################################
        ########## 3. STOCK TREND DATA ############################################
        ############################################################################
        merged_stock_trend_df_ = pd.read_csv(self.stock_trend_path, header=None)
        merged_stock_trend_df = torch.tensor(merged_stock_trend_df_.values).unsqueeze(0)
        # print(merged_stock_trend_df.shape)
        train_num = (train_fin_idx - seq) * 1
        valid_num = (val_fin_idx - train_fin_idx) * 1
        test_num = (len(merged_stock_trend_df_) - val_fin_idx) * 1

        
        stock_trend_train = np.zeros([train_num, seq, self.num_stocks], dtype=float)
        ########### Train Data ##################
        ins_ind=0
        for idx in range(start_idx, train_fin_idx):
            for tic_idx in range(1):
                stock_trend_train[ins_ind] = merged_stock_trend_df[tic_idx][idx-seq: idx, : ]
                # stock_trend_train[ins_ind] = merged_stock_trend_df[idx-seq: idx, : ]
        
                ins_ind+=1
               
        ########## Valid Data ####################
        ins_ind=0
        stock_trend_valid = np.zeros([valid_num, seq, self.num_stocks], dtype=float)
                
        # for idx in range(train_end, valid_end):
        for idx in range(train_fin_idx, val_fin_idx):
            for tic_idx in range(1):
                stock_trend_valid[ins_ind] = merged_stock_trend_df[tic_idx][idx-seq: idx, :]
                # stock_trend_valid[ins_ind] = merged_stock_trend_df[idx-seq: idx, :]
                
                ins_ind+=1

        ############ TEST Data #####################
        ins_ind=0
        stock_trend_test = np.zeros([test_num, seq, self.num_stocks], dtype=float)   
        
        for idx in range(val_fin_idx, end):
            # for idx in range(valid_end, test_end):
            for tic_idx in range(1):
                stock_trend_test[ins_ind] = merged_stock_trend_df[tic_idx][idx - seq: idx, : ]
                # stock_trend_test[ins_ind] = merged_stock_trend_df[idx - seq: idx, : ]
                
                ins_ind+=1


        print(price_train.shape)
        print(keyword_train.shape)
        print(stock_trend_train.shape)


        ################ RESULT #################################################
        train_x = [price_train, keyword_train, stock_trend_train]
        train_y = price_target

        valid_x = [price_valid, keyword_valid, stock_trend_valid]
        valid_y = target_valid

        test_x = [price_test, keyword_test, stock_trend_test]
        test_y = target_test

        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def create_data(self, batch_size, train_x, train_k, train_s, train_y, 
    valid_x, valid_k, valid_s, valid_y, 
    test_x, test_k, test_s, test_y):
        #make tensors of 91 items, and make a list of those tensors
        fin_train_x=[]
        fin_train_k=[]
        fin_train_s=[]
        fin_train_y=[]
        
        fin_valid_x=[]
        fin_valid_k=[]
        fin_valid_s=[]
        fin_valid_y=[]
        
        fin_test_x=[]
        fin_test_k=[]
        fin_test_s=[]
        fin_test_y=[]

        #############TRAIN #####################
        for i in range(0, len(train_x), self.num_stocks):
            one_x = train_x[i:i+self.num_stocks]
            one_y = train_y[i:i+self.num_stocks]
            fin_train_x.append(one_x.squeeze())
            fin_train_y.append(one_y.squeeze())

        for i in range(0, len(train_k), len(self.keywords_list)):
            one_k = train_k[i:i+len(self.keywords_list)]
            fin_train_k.append(one_k.squeeze())

        for i in range(0, len(train_s)):
            one_s = train_s[i]
            fin_train_s.append(one_s.squeeze())

        #############VALID #####################
        for i in range(0, len(valid_x), self.num_stocks):
            one_x = valid_x[i:i+self.num_stocks]
            one_y = valid_y[i:i+self.num_stocks]
            fin_valid_x.append(one_x.squeeze())
            fin_valid_y.append(one_y.squeeze())

        for i in range(0, len(valid_k), len(self.keywords_list)):
            one_k = valid_k[i:i+len(self.keywords_list)]
            fin_valid_k.append(one_k.squeeze())

        for i in range(0, len(valid_s)):
            one_s = valid_s[i]
            fin_valid_s.append(one_s.squeeze())

        ########### TEST  ################
        for i in range(0, len(test_x), self.num_stocks):
            one_x = test_x[i:i+self.num_stocks]
            one_y = test_y[i:i+self.num_stocks]

            fin_test_x.append(one_x.squeeze())
            fin_test_y.append(one_y.squeeze())

        for i in range(0, len(test_k), len(self.keywords_list)):
            one_k = test_k[i:i+len(self.keywords_list)]
            fin_test_k.append(one_k.squeeze())

        for i in range(0, len(test_s)):
            one_s = test_s[i]
            fin_test_s.append(one_s.squeeze())

        print(torch.tensor(fin_train_x).shape)
        print(torch.tensor(fin_train_k).shape)
        print(torch.tensor(fin_train_s).shape)
        print(torch.tensor(fin_train_y).shape)

        train_x = []
        valid_x=[]
        test_x=[]
        for i in range(len(fin_train_x)):
            train_x.append([torch.FloatTensor(fin_train_x[i][:,:,:]), torch.FloatTensor(fin_train_k[i][:,:,:]),torch.FloatTensor(fin_train_s[i][:,:])])
        for i in range(len(fin_valid_x)):
            valid_x.append([torch.FloatTensor(fin_valid_x[i][:,:,:]), torch.FloatTensor(fin_valid_k[i][:,:,:]),torch.FloatTensor(fin_valid_s[i][:,:])])
        for i in range(len(fin_test_x)):
            test_x.append([torch.FloatTensor(fin_test_x[i][:,:,:]), torch.FloatTensor(fin_test_k[i][:,:,:]),torch.FloatTensor(fin_test_s[i][:,:])])
        
        
        # train_dataset = CustomDataset(fin_train_x, fin_train_y)
        train_dataset = CustomDataset(train_x, fin_train_y)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # valid_dataset = CustomDataset(fin_valid_x, fin_valid_y)
        valid_dataset = CustomDataset(valid_x, fin_valid_y)
        validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        # test_dataset = CustomDataset(fin_test_x, fin_test_y)
        test_dataset = CustomDataset(test_x, fin_test_y)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return trainloader, validloader, testloader

class CustomDataset(Dataset): 
  def __init__(self, x_data, y_data):
    self.x_data = x_data
    self.y_data = y_data

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 

    # x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    x = self.x_data[idx]
    # y= self.y_data[idx]
    return x, y

