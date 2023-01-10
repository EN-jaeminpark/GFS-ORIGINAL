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

class make_trend_graph:
    def __init__(self, data_num) -> None:
        self.companies = os.listdir('./Dataset_' + str(data_num) + '/Final/Price/')

    def run(self, trends_tensor, k=5):
        # Make center nodes and conne/ct nodes to the centers
        
        ######### GRAPH INFO ##########
        edge_list_from=[]
        edge_list_to=[]
        weights=[]
        x=[]
        ###############################

        ######### CORR CALCULATE ######
        corr_matrix = []
        corr_scores = []
        # print('EACH ITEM SHAPE')
        # print(trends_tensor.shape)
        
        for idx1 in range(trends_tensor.size(1)):
            corr_score = 0
            corr_row = []

            trends1 = trends_tensor[:, idx1]
            x.append([trends1])

            for idx2 in range(trends_tensor.size(1)):
                if idx1 == idx2:
                    edge_list_from.append(idx1)
                    edge_list_to.append(idx2)
                    weights.append(1)
                    
                    corr = 0
                else:
                    trends2 = trends_tensor[:, idx2]
                    corr = self.correlation_metric(trends1, trends2)

                if idx1 != idx2 and abs(corr) > 0.8:
                    edge_list_from.append(idx1)
                    edge_list_to.append(idx2)
                    weights.append(abs(corr))
                    
                corr_score+=abs(corr)
                
                corr_row.append(abs(corr))

            corr_scores.append([idx1, corr_score])
            corr_matrix.append(corr_row)

        #################################

        ######## FIND K CENTERS #########
        sorted_k = sorted(corr_scores, key=lambda item:item[1], reverse=True)
        top_k = sorted_k[:k]
        centers = [x[0] for x in top_k]
        # print(centers)
        #################################
        
        ######## Create Edges Between TOP Ks #########
        for i in centers:
            for j in centers:
                if i != j:
                    edge_list_from.append(i)
                    edge_list_to.append(j)
                    weights.append(1)
        ###############################################
        # print(centers)
        ######## Connect nodes to Centers #############
        for idx, row in enumerate(corr_matrix):
            if idx in centers:
                continue
            else:
                
                corr_to_centers = [row[center] for center in centers]
                
                max_corr = max(corr_to_centers)                
                max_center_idx = np.argmax(corr_to_centers)
                # max_center_idx = corr_to_centers.index(max_corr)
                max_center = centers[max_center_idx]
                
                edge_list_from.append(idx)
                edge_list_to.append(max_center)
                weights.append(max_corr)
        ###############################################    

        edge_index = [edge_list_from, edge_list_to]
       

        return edge_index, weights, x

    def correlation_metric(self, y_pred, y_true):
        x = torch.tensor(y_pred)
        y = torch.tensor(y_true)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        return corr


def make_stationary_graph(data_num):
    data_num = str(data_num)
    graph_folder = './Dataset_'+ data_num+'/'
    companies = os.listdir('./Dataset_' + data_num+'/Final/Price/')
    company_list = [x.split('.')[0] for x in companies]
    edge_from =[]
    edge_to=[]
    weights=[]
    x=[]

    comp_df= pd.DataFrame()
    comps =[]
    cntrys=[]
    sects=[]
    inds = []
    for idx, company in enumerate(company_list):

        comp = yf.Ticker(company)
  
        sector = comp.info['sector']
        industry = comp.info['industry']
        country = comp.info['country']
        
        comps.append(company)
        cntrys.append(country)
        sects.append(sector)
        inds.append(industry)
    
    if len(comps) == len(cntrys) and len(cntrys) == len(sects):
        comp_df['Ticker'] = comps
        comp_df['Country'] = cntrys
        comp_df['Sector'] =sects
        comp_df['Industry'] = inds 
    
    for idx1, comp in enumerate(company_list):
        index = comp_df.index[comp_df.Ticker == comp].tolist()[0]
        cntry = comp_df['Country'][index]
        sect = comp_df['Sector'][index]
        ind = comp_df['Industry'][index]

        for idx2, comp2 in enumerate(company_list):
            weight=0
            index2 = comp_df.index[comp_df.Ticker == comp2].tolist()[0]
            cntry2 = comp_df['Country'][index2]
            sect2 = comp_df['Sector'][index2]
            ind2 = comp_df['Industry'][index2]

            if cntry == cntry2:
                weight+=0.2
            if sect == sect2:
                weight+=0.4
            if ind == ind2:
                weight+=0.4
            
            if weight > 0:
                edge_from.append(idx)
                edge_to.append(idx2)
                weights.append(weight)
    
    edge_index = [edge_from, edge_to]
    final = [edge_index, weights, x]
    with open(graph_folder+'stationary_graph.pkl', 'wb') as f:
        pass
    with open(graph_folder+'stationary_graph.pkl', 'wb') as f:
        pickle.dump(final, f)      

