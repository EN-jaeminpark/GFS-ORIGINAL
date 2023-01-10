import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
random_seed = 77
np.random.seed(random_seed)
torch.manual_seed(random_seed)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Train:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, epoch, device, g2_ctx, model_path,
                 train_graphs, valid_graphs, scheduler):
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epoch
        self.device = device
        self.g2_ctx = g2_ctx
        self.model_path = model_path
        self.train_graphs = train_graphs
        self.valid_graphs = valid_graphs
        self.scheduler = scheduler

    def start(self):   
        best_valid_loss = 5
        best_acc=0
        torch.autograd.set_detect_anomaly(True)
    
        for epoch_num in tqdm(range(self.epoch)):
            
            epoch_loss = 0
            epoch_acc = 0
            epoch_mcc = 0

            ############## TRAIN #############################################
            self.model.train()
            for batch_num, data in enumerate(self.train_loader, 0):

                inputs, labels = data
                price_data = inputs[0].to(self.device)
                keyword_trend_data = inputs[1].float().to(self.device)
                labels = labels.to(self.device)

                batch_loss = 0
                batch_acc= 0
                batch_mcc = 0

                # print(price_data.shape)
                # print(keyword_trend_data.shape)
                

                self.model.zero_grad()
                self.optimizer.zero_grad()
                with torch.autograd.detect_anomaly():
                    for item_num in range(price_data.size(0)):
                        
                        price = price_data[item_num].to(self.device)
                        keytrend = keyword_trend_data[item_num].to(self.device)
                        
                        item_label = labels[item_num]
                        item_label = item_label.type(torch.LongTensor).to(self.device)
                        
                        output = self.model.forward(price,keytrend, self.g2_ctx, 
                        self.train_graphs[batch_num][item_num])
                        
                        loss = self.criterion(output.squeeze(), item_label.float())      
                        acc, mcc= self.acc_calc2(output, item_label)                
                        
                        # print(acc, mcc)
                        
                        batch_loss += loss
                        batch_acc += acc
                        batch_mcc += mcc
                        # loss.backward()
                        # self.optimizer.step()
                        
                    batch_loss.backward()
                    self.optimizer.step()
                    # batch_loss.backward()
                    
                print('Train Loss for Batch {0} is {1}'.format(batch_num, batch_loss / price_data.size(0)))
                print('Train ACC for Batch {0} is {1}'.format(batch_num, batch_acc / price_data.size(0)))
                print('Train MCC for Batch {0} is {1}'.format(batch_num, batch_mcc / price_data.size(0)))

                bl = batch_loss / price_data.size(0)
                ba = batch_acc / price_data.size(0)
                bm = batch_mcc / price_data.size(0)

                epoch_loss += bl
                epoch_acc += ba
                epoch_mcc += bm
                
            self.scheduler.step()
            
            final_loss = epoch_loss / len(self.train_loader)
            final_acc = epoch_acc / len(self.train_loader)       
            final_mcc = epoch_mcc / len(self.train_loader)
            
            print(' ------------- Train Step ------------------')
            print(' Train Loss for Epoch {0} is {1}'.format(epoch_num, final_loss))
            print(' Train ACC for Epoch {0} is {1}'.format(epoch_num, final_acc))
            print(' Train MCC for Epoch {0} is {1}'.format(epoch_num, final_mcc))
            print(' ------------- ----------- ------------------')


            ################################ VALIDATION ######################
            self.model.eval()
            epoch_loss = 0
            epoch_acc = 0
            epoch_mcc = 0
            for batch_num, data in enumerate(self.valid_loader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                batch_loss = 0
                batch_acc= 0
                batch_mcc = 0
                with torch.no_grad():
                    for item_num in range(inputs.size(0)):
                        
                        item = inputs[item_num]
                        item_label = labels[item_num]
                        item_label = item_label.type(torch.LongTensor).to(self.device)

                        
                        output = self.model.forward(item, self.g2_ctx, self.valid_graphs[batch_num][item_num])                                           
                        loss = self.criterion(output.squeeze(), item_label.squeeze(dim=-1).float())
                        acc, mcc= self.acc_calc2(output, item_label)      
                        
                        batch_loss += loss.item()
                        batch_acc += acc
                        batch_mcc += mcc
                    
                    print(' Validation Loss for Batch {0} is {1}'.format(batch_num, batch_loss / inputs.size(0)))
                    print(' Validation ACC for Batch {0} is {1}'.format(batch_num, batch_acc / inputs.size(0)))
                    print(' Validation MCC for Batch {0} is {1}'.format(batch_num, batch_mcc / inputs.size(0)))
  

                    bl = batch_loss / inputs.size(0)
                    ba = batch_acc / inputs.size(0)
                    bm = batch_mcc / inputs.size(0)

                    epoch_loss += bl
                    epoch_acc += ba
                    epoch_mcc += bm
                
            final_loss_ = epoch_loss / len(self.valid_loader)
            final_acc_ = epoch_acc / len(self.valid_loader)  
            final_mcc_ = epoch_mcc / len(self.valid_loader)               
            
            print(' ------------- Validation Step  ------------------')
            print(' Validation Loss for Epoch {0} is {1}'.format(epoch_num, final_loss_))
            print(' Validation ACC for Epoch {0} is {1}'.format(epoch_num, final_acc_))
            print(' Validation MCC for Epoch {0} is {1}'.format(epoch_num, final_mcc_))

            if float(final_acc_) > best_acc:
                best_acc = float(final_acc_)
                print('**************** Best Model saved **************')
                torch.save(self.model.state_dict(), self.model_path)

        #####################################################      
        print('Best Model saved at path: ', self.model_path)            
        print('Best Valid ACC: ', best_acc)
        #####################################################
    
    def acc_calc(self, pred, label):
        acc = 0
        pred_list=[]
        for idx, item in enumerate(pred): 
            if item > 0.5:
                p=1
                pred_list.append(p)
            else:
                p=0
                pred_list.append(p)

            if p == label[idx]:
                acc +=1
            else:
                acc += 0
        
        ac = acc / len(pred)
        return ac 

    def acc_calc2(self, pred, label):
        acc = 0
        label = label.cpu().detach().numpy()
        predictions = []
        for idx, item in enumerate(pred):
            if item > 0.5:
                p = 1
            else:
                p = 0
            predictions.append(p)

        predictions = np.array(predictions)
        
        acc = accuracy_score(predictions, label)
        mcc = matthews_corrcoef(predictions, label)
        # print(acc)
        # print(mcc)
        return acc, mcc

