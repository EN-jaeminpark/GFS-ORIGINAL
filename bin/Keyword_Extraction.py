import re
from tqdm import tqdm
import numpy as np
import time
from threading import Thread
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pickle
import sys
import csv
# csv.field_size_limit(sys.maxsize)
csv.field_size_limit(100000000)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# nltk.download('stopwords')
# nltk.download('punkt')

#################################################################################################
#semicon focused
#companies: intel, nvidia, texas instrument, micron, analog, microchip, skyworks, AMD, TSMC, samsung, sk hynix, qualcomm

class Extractor():
    def __init__(self):
        self.list_of_companies=[]
        self.news_stopwords = []
        
    def Main(self):
        print('File Extraction Start...')
        self._read_stopwords
        
        #If keyword extraction on companies
        #####CHECK PLS###########
        # article_folders = './News_archive/'

        #If keword extraction on sectors
        article_folders = './News_Sectors/'

        # self._tokenize_articles(article_folders)
        print('tokenization complete')
        
        if len(self.list_of_companies)<1:
            self.list_of_companies = os.listdir('./News_Sectors')

        # self._make_combined_dict_for_sf()
        for comp in self.list_of_companies:
            print(comp)
            self._extract_keywords_for_each_company(comp)
        
        
    def _read_stopwords(self):
        f = open('news_stopwords.txt', 'r')
        self.news_stopwords = f.readline()
        f.close()
    
    def _read_csv_and_merge_news_files(self, company_name, news_file_paths):
        final_csv = []

        for news_file in news_file_paths:
            news_csv = pd.read_csv(news_file, index_col=0, engine='python')
            
            if 'Data' in news_csv.columns:
                news_csv = news_csv.drop('Data', axis=1)
            
            for i in range(len(news_csv)):
                date = news_csv['Date'][i]
                article = news_csv['Article'][i]
                date_article_pair = [date, article]
                final_csv.append(date_article_pair)
          
            print('Total articles: ', len(final_csv))

            
            # print(final_csv.shape)
            with open('article_pickles/'+company_name+'.pkl','wb') as f:
                pass
            with open('article_pickles/'+company_name+'.pkl','wb') as f:
                pickle.dump(final_csv,f)
            
            # print('dd')

        return final_csv
        
    def filter_words(self, word_tokens):
        stop_words = set(stopwords.words('english'))
        lower_words = [w.lower() for w in word_tokens]
            
        filtered_words=[]
        for word in lower_words:
            if word not in stop_words:
                if word not in self.news_stopwords:
                    if len(word) > 1:
                        if word != 'nan':
                            filtered_words.append(word)
                        else:
                            continue
                    else:
                        continue
                else:
                    continue                            
            else:
                continue
        return filtered_words
  
    def _edit_date(self, date):
        if 'PUBLISHED' or 'UPDATED' in date:
            #its from CNBC!
            try: 
                tmp = date.split(',')
                date_time = str(tmp[1])
                # print(date_time)
                date_only = date_time.split(':')
                date_ = date_only[0][:-1].split(' ')
                year = str(date_[3])
                month_ = str(date_[1])
                month = self._change_str_date_to_num(month_)
                day = str(date_[2])
                return year +'-'+month+'-'+day

            except IndexError:
                # CNN
                date = date.split(',')
                year = date[1]
                m_d = date[0].split(' ')
                month = m_d[0]
                month = self._change_str_date_to_num(month)
                day = m_d[1]
                return year +'-'+month+'-'+day

            except:
                print(date)
                pass            

        else:
            print(date)
            print('not yet developed')
            
    def _change_str_date_to_num(self, month):
        if month =='JAN' or month =='January':
            m = '01'
        elif month =='FEB' or month =='February':
            m = '02'
        elif month =='MAR' or month == 'March':
            m = '03'
        elif month =='APR' or month =='April':
            m = '04'
        elif month =='MAY' or month =='May':
            m = '05'
        elif month =='JUN' or month =='June':
            m = '06'
        elif month =='JUL' or month == 'July':
            m = '07'
        elif month =='AUG' or month == 'August':
            m = '08'
        elif month =='SEP' or month == 'September':
            m = '09'
        elif month =='OCT' or month =='October':
            m = '10'
        elif month =='NOV' or month =='November':
            m = '11'
        elif month =='DEC' or month =='December':
            m = '12'
        else:
            m = month
        return m 
                       
    def _tokenize_articles(self, articles_folder_path):
        base_folder = articles_folder_path
        
        #contains names of all the companies
        company_folders = os.listdir(base_folder)
        
        for company_folder in company_folders:
            company_name = company_folder
            print('Tokenization for company: ', company_name)

            self.list_of_companies.append(company_name)
            company_folder_path = base_folder + company_name +'/'
            
            news_files = os.listdir(company_folder_path)
            news_file_paths = [company_folder_path + w for w in news_files]
            
            articles_combined = self._read_csv_and_merge_news_files(company_name, news_file_paths)
            
            
            preprocessed_list =[]

            for article_pair in articles_combined:
                
                date_tmp = article_pair[0]
                date = self._edit_date(date_tmp)

                article = article_pair[1]
                
                preprocessed_text = re.sub('[^a-zA-Z]', ' ', str(article))
                word_tokens = word_tokenize(preprocessed_text)
                filtered_words = self.filter_words(word_tokens)
                
                filtered_dict=dict(Counter(filtered_words))          
                    
                preprocessed_list.append([date, filtered_dict])
                            
            with open('News_tokenized/PKL_DICT/'+company_name+'.pkl','wb') as f:
                pass
            with open('News_tokenized/PKL_DICT/'+company_name+'.pkl','wb') as f:
                pickle.dump(preprocessed_list,f)

    def _make_combined_dict_for_sf(self):
        base_path = 'News_tokenized/PKL_DICT/'
        for company in self.list_of_companies:
            pkl_path = base_path + company +'.pkl'
            
            with open(pkl_path, 'rb') as f:
                articles = pickle.load(f)
            
            total_tokens=[]
            for article in articles:
                keys = article[1].keys()
                for key in keys:
                    if key not in total_tokens:
                        total_tokens.append(key)
            
            new_dict={}
            
            for token in tqdm(total_tokens):
                token_count = 0
                
                for article in articles:
                    tok_num = article[1].get(token,0)
                    token_count+=tok_num
                    
                new_dict[token] = token_count    
            
            with open(base_path+company+'_total.pkl', 'wb') as f:
                pass
            with open(base_path+company+'_total.pkl', 'wb') as f:
                pickle.dump([new_dict],f)

    def _extract_keywords_for_each_company(self, company):
        #read tokenized pkl!
        base_path = 'News_tokenized/PKL_DICT/'
        pkl_file = base_path+company + '.pkl'
        with open(pkl_file, 'rb') as f:
            articles = pickle.load(f)
        
        
        #나중에 지우기        
        total_tokens=[]
        for article in tqdm(articles):
            keys = article[1].keys()
            for key in keys:
                if key not in total_tokens:
                    total_tokens.append(key)

        top50=[]
        print('Extracting tokens from comapny: ', company) 
        #Used in SF calc
        number_of_companies = len(self.list_of_companies)
                
        for token in tqdm(total_tokens):
            TF_SCORE = 0
            IDF_SCORE = 0

            
            SF_SCORE = 0
                    
            document_count = 0
            num_articles = len(articles)
            
            for article in articles:
                # token_count = 0
                article_length = len(article[1])
                
                num_tokens = article[1].get(token, 0)
                TF_SCORE += (num_tokens / (article_length + 1))

                if num_tokens != 0:
                    document_count+=1
            
            TF_SCORE = TF_SCORE / num_articles      
            IDF_SCORE = num_articles / (document_count + 1)
            
            #SF score
            stock_count = 0
            for other_company in self.list_of_companies:
                if other_company == company:
                    continue
                total_pkl_path = base_path + other_company+'_total.pkl'
                with open(total_pkl_path, 'rb') as f:
                    total_pkl = pickle.load(f)
                
            
                total_dict = total_pkl[0]
                freq = total_dict.get(token,0)
                if freq > 5:
                    stock_count+=1
            
            SF_SCORE = number_of_companies / (stock_count + 1)
            
            final_score = np.log(TF_SCORE) + np.log(IDF_SCORE) + np.log(SF_SCORE)
            # print(TF_SCORE, IDF_SCORE, SF_SCORE)
            # print(final_score)
            
            score_set = [token, final_score]
            
            if len(top50) <50:
                top50.append(score_set)
            else:
                sorted_list = sorted(top50, key=lambda item: item[1], reverse = True)
                if final_score > sorted_list[-1][1]:
                    top50.remove(sorted_list[-1])
                    top50.append(score_set)
            
        #SCORING COMPLETE
        sorted_list = sorted(top50, key=lambda item: item[1], reverse = True)
        print('---------------------- Top10 Keywords for company: ', company, ' -------------------------')
        print(sorted_list[:10])
        print('------------------------------------------------------------------------------------------')
        with open('Results/'+company+'_top50.pkl','wb') as f:
            pass
        with open('Results/'+company+'_top50.pkl','wb') as f:
            pickle.dump(top50, f)
                
                
if __name__ == '__main__':
    KE = Extractor()
    KE.Main()
    # with open('News_tokenized/PKL_DICT/adobe_total.pkl','rb') as f:
    #     a = pickle.load(f)
    # print(a[:30])
    
    



























