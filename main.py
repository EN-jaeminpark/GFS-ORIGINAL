from model.GFS import *
from utils.load_data import *
from utils.load_graph import *
from utils.trainer import *
from utils.tester import *
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
import math
import random
import argparse

random_seed = 77
np.random.seed(random_seed)
torch.manual_seed(random_seed)

#################################### ARGPARSE #######################################
parser = argparse.ArgumentParser(description='GFS')

#dataset 1: 2019, 2: 2020, 3: 2021
parser.add_argument('--n', default='1', type=int, help='dataset_num')

parser.add_argument('--epoch', default=100, type=int, help='num epochs')
parser.add_argument('--seq', default=10, type=int, help='num epochs')
parser.add_argument('--gpu', default='cuda', type=str, help='num epochs')
parser.add_argument('--batch', default=16, type=int, help='num epochs')
parser.add_argument('--lr', default=0.001, type=float, help='num epochs')
parser.add_argument('--best_model', default='./Params/20_model.pt', type=str)
parser.add_argument('--mode', default='train', type=str, help='mode')
parser.add_argument('--graph', default='no', type=str, help='mode')
parser.add_argument('--train_start', default=0, type=int, help='train starting idx')
parser.add_argument('--valid_start', default=95, type=int, help='valid start idx')
parser.add_argument('--test_start', default=169, type=int, help='test start idx')
parser.add_argument('--end', default=222, type=int, help='test end idx. no need to enter')
args = parser.parse_args()
#################################### ######### #######################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
def make_corr_graphs(dataloader, purpose, SEQ, batch, end):
    
    final_data = []
    print('Process for... : ', purpose)

    for batch_num, data in enumerate(tqdm(dataloader)):
        inputs, labels = data

        batch_data = []
        
        trends = inputs[2]

        # print('FROM LOADER DIRECT SIZE')
        # print(trends.shape)
        
        for item_num in range(trends.size(0)):
            # item = inputs[item_num]
            
            #all are the same. so use the one from the first stock
            
            x_ = []

            GL = make_trend_graph(args.n)
            edge_index, weights, x = GL.run(trends[item_num])
            batch_data.append([edge_index, weights, x_])
            
        final_data.append(batch_data)
    
    with open('./Graphs/'+str(args.n) + purpose+'_graphs_'+str(SEQ)+'_'+str(batch)+'_'+str(end)+'.pkl', 'wb') as f:
        pass
    with open('./Graphs/'+str(args.n)+purpose+'_graphs_'+str(SEQ)+'_'+str(batch)+'_'+str(end)+'.pkl', 'wb') as f:
        pickle.dump(final_data, f)
    
    return final_data     


if __name__ == '__main__':

    ############ HYPERPARAMS ###############################################
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print('Current Device: ', device)
    SEQ = args.seq
    LR = args.lr
    NUM_EPOCH = args.epoch
    batch = args.batch
    
    ts = args.train_start
    vs = args.valid_start
    tests = args.test_start
    end = args.end
    #########################################################################

    ############### DATALOADER ##############################################
    price_folder ='./Dataset_'+ str(args.n) +'/Final/Price/'
    stock_trend_path ='./Dataset_' + str(args.n) +'/Final/Trend/stocks_merged.csv'
    keywords_trend_folder ='./Dataset_' + str(args.n) +'/Final/Keywords/'

    DL = Dataset_Loader(price_folder, stock_trend_path, keywords_trend_folder)
    trainloader, validloader, testloader = DL.run(seq=SEQ, batch_size = batch, train_start=ts, valid_start=vs, 
                                                  test_start=tests, end=end)
    #########################################################################

    #### Define model, optimizer, loss #######################################
    Model = GFS(device, feature_dim=16, price_dim =10, trend_dim =7, seq=SEQ).to(device)
    # optimizer = torch.optim.SGD(Model.parameters(), lr=LR, momentum=0.8)
    optimizer = torch.optim.Adam(Model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.9 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
    print(f'The model has {count_parameters(Model):,} trainable parameters')
    Model.apply(initialize_weights)
    ###################################################################################

    ###################### LOAD GRAPHS #################################################
    # GL = load_graph()
    stationary_graph_path = './Dataset_' +str(args.n) +'/stationary_graph.pkl'
    if not os.path.exists(stationary_graph_path):
         make_stationary_graph(args.n)
    with open(stationary_graph_path, 'rb') as f:
        edge_index, edge_attr, x = pickle.load(f)
    # edge_index, edge_attr, x = GL.stationary_graph()

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(x, dtype=torch.float)
    
    G2 = [edge_index, edge_attr, x]   
    
    if not os.path.exists('./Graphs/'+str(args.n) +'test_graphs_'+str(SEQ)+'_'+str(batch)+'_'+str(end)+'.pkl'):
        print('Creating GTREND GRAPHS...THIS MIGHT TAKE A WHILE')
        
        final_data_for_train_loader = make_corr_graphs(trainloader, 'train', SEQ, batch, end)
        print('Train Graph fin')
        
        final_data_for_valid_loader = make_corr_graphs(validloader, 'valid', SEQ, batch, end)
        print('Valid Graph fin')

        final_data_for_test_loader = make_corr_graphs(testloader, 'test', SEQ, batch, end)      
        print('Test Graph fin')
        
    else:
        with open('./Graphs/'+str(args.n) +'train_graphs_'+str(SEQ)+'_'+str(batch)+'_'+str(end)+'.pkl', 'rb') as f:
            final_data_for_train_loader = pickle.load(f)
        with open('./Graphs/'+str(args.n) +'valid_graphs_'+str(SEQ)+'_'+str(batch)+'_'+str(end)+'.pkl', 'rb') as f:
            final_data_for_valid_loader = pickle.load(f)
        with open('./Graphs/'+str(args.n) +'test_graphs_'+str(SEQ)+'_'+str(batch)+'_'+str(end)+'.pkl', 'rb') as f:
            final_data_for_test_loader = pickle.load(f)

    #####################################################################################

    ########## TRAIN & TEST ###############################################################
    if args.mode == 'train':
        model_save_path = './Params/dataset'+str(args.n)+'_'+str(args.epoch)+'_'+str(args.lr)+'_'+str(args.seq)+'_'+str(args.batch)+'_'+str(end)+'_model.pt'
        # Trainer = Train(Model, trainloader, validloader, optimizer, criterion,
        # NUM_EPOCH, device, G2, model_save_path, final_data_for_train_loader, final_data_for_valid_loader, scheduler)
        Trainer = Train(Model, trainloader, validloader, optimizer, criterion,
        NUM_EPOCH, device, G2, model_save_path, final_data_for_train_loader, final_data_for_valid_loader, scheduler)
        Trainer.start()
    
    if args.mode == 'test':
        best_model_path = args.best_model
        test_model =GFS(device, feature_dim=16, price_dim = 10, trend_dim = 7, seq=SEQ).to(device)
        test_model.load_state_dict(torch.load(best_model_path, map_location=device))
        Tester = Test(test_model, testloader, criterion, device, G2, final_data_for_test_loader)
        Tester.Start()
    ######################################################################################
        
    




