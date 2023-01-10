import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from statistics import mode
from collections import Counter
random_seed = 77
np.random.seed(random_seed)
torch.manual_seed(random_seed)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Test:
    def __init__(self, model, test_loader, criterion, device, g2_ctx, test_graphs):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.g2_ctx = g2_ctx
        self.test_graphs = test_graphs

    def Start(self):
        epoch_loss = 0
        epoch_acc = 0
        epoch_mcc = 0
        low_list=[]
        self.model.eval()
        wrong_list_composite = []
        tot_item=0
        
        for batch_num, data in enumerate(self.test_loader, 0):

            inputs, labels = data
            price_data = torch.FloatTensor(inputs[0]).to(self.device)
            keyword_trend_data = inputs[1].float().to(self.device)
            labels = labels.to(self.device)

            batch_loss = 0
            batch_acc= 0
            batch_mcc = 0

    
            with torch.no_grad():
                for item_num in range(price_data.size(0)):
                    
                    price = price_data[item_num].to(self.device)
                    keytrend = keyword_trend_data[item_num].to(self.device)
                    
                    item_label = labels[item_num]
                    item_label = item_label.type(torch.LongTensor).to(self.device)
                    
                    output = self.model.forward(price,keytrend, self.g2_ctx, 
                    self.test_graphs[batch_num][item_num])
                    # print(output)
                    
                    loss = self.criterion(output.squeeze(), item_label.float())      
                    acc, mcc, wrong_list= self.acc_calc(output, item_label)                
                    wrong_list_composite =wrong_list_composite + wrong_list
                    # print(acc, mcc)
                    
                    batch_loss += loss
                    batch_acc += acc
                    batch_mcc += mcc
                    
                    epoch_loss+=loss
                    epoch_acc+=acc
                    epoch_mcc +=mcc
                    tot_item+=1
        
                
            print('Test Loss for Batch {0} is {1}'.format(batch_num, batch_loss / price_data.size(0)))
            print('Test ACC for Batch {0} is {1}'.format(batch_num, batch_acc / price_data.size(0)))
            print('Test MCC for Batch {0} is {1}'.format(batch_num, batch_mcc / price_data.size(0)))
            
            bl = batch_loss / price_data.size(0)
            ba = batch_acc / price_data.size(0)
            bm = batch_mcc / price_data.size(0)

            # epoch_loss += bl
            # epoch_acc += ba
            # epoch_mcc += bm
        
        final_loss = epoch_loss / tot_item
        final_acc = epoch_acc / tot_item
        final_mcc = epoch_mcc / tot_item       
        
        # print(wrong_list_composite)
        c = Counter(wrong_list_composite)
        
        
        print(' ------------- TEST RESULT  ------------------')
        print(' ------------- -----------  ------------------')
        print(' Final TEST Loss is... {0}'.format(final_loss))
        print(' Final TEST ACC is... {0}'.format(final_acc))
        print(' Final TEST MCC is... {0}'.format(final_mcc))
        print(' ------------- -----------  ------------------')
        print(' ------------- -----------  ------------------')
        
        print('------------------------------------------------')
        print(c.most_common(10))
        print('------------------------------------------------')

    def acc_calc(self, pred, label):
        acc = 0
        label = label.cpu().detach().numpy()
        predictions = []
        wrong_list = []
        for idx, item in enumerate(pred):
            if item >= 0.5:
                p = 1
            else:
                p = 0
                
            predictions.append(p)
            
            if p == label[idx]:
                pass
                # acc +=1
            else:
                wrong_list.append(idx)
                # acc += 0

        predictions = np.array(predictions)
        print(predictions)
        accu = accuracy_score(predictions, label)
        mcc = matthews_corrcoef(predictions, label)
                
        ac = acc / len(pred)
        return accu, mcc, wrong_list
            