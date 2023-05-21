import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ANN_models import PredictionModel, PredictionModel2
from myDataset import myDataset
import numpy as np
import pandas as pd
import time
import json
import os
import sys
os.chdir(sys.path[0])

with open(os.path.join("OPs_used", "bloodConc.json"), "r") as f:
    bloodConc = json.load(f)
    bloodConc = bloodConc['bloodConc']
with open(os.path.join("OPs_used", "SO2.json"), 'r') as f:
    SO2 = json.load(f)
    train_SO2 = SO2['train_SO2']
    test_SO2 = SO2['test_SO2']
with open(os.path.join('OPs_used', "muscle_SO2.json"), 'r') as f:
    muscle_SO2 = json.load(f)
    muscle_SO2 = muscle_SO2['SO2']

#%% train model
def train(model, optimizer, criterion, train_loader, test_loader, epoch, batch_size, lr):
    trlog = {}
    trlog['epoch'] = epoch
    trlog['batch_size'] = batch_size
    trlog['learning_rate'] = lr
    trlog['train_loss'] = []
    trlog['test_loss'] = []
    min_loss = 100000
    for ep in range(epoch):
        model.train()
        tr_loss = 0
        for batch_idx, (data,target,_,_) in enumerate(train_loader):
            data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
            optimizer.zero_grad()
            output = model(data)
            output = output
            target = target
            loss = criterion(output,target)
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % int(0.1*len(train_loader)) == 0:
                print(f"[train] ep:{ep}/{epoch}({100*ep/epoch:.2f}%) batch:{batch_idx}/{len(train_loader)}({100*batch_idx/len(train_loader):.2f}%)\
                      loss={tr_loss/(batch_idx+1)}")
        trlog['train_loss'].append(tr_loss/len(train_loader))
        min_loss = train_test(trlog,ep,min_loss, test_loader)
        
    
    return trlog

def train_test(trlog,ep,min_loss, test_loader):
    model.eval()
    ts_loss = 0
    for batch_idx, (data,target,_,_) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        optimizer.zero_grad()
        output = model(data)
        output = output
        target = target
        loss = criterion(output,target)
        ts_loss += loss.item()
        
    print(f"[test] batch:{batch_idx}/{len(test_loader)}({100*batch_idx/len(test_loader):.2f}%) loss={ts_loss/len(test_loader)}")
    trlog['test_loss'].append(ts_loss/len(test_loader))
    
    if min_loss > ts_loss/len(test_loader):
        min_loss = ts_loss/len(test_loader)
        trlog['best_model'] = os.path.join("model_save",result_folder,f"ep_{ep}_loss_{min_loss}.pth")
        torch.save(model.state_dict(), os.path.join("model_save",result_folder,f"ep_{ep}_loss_{min_loss}.pth"))
            
    return min_loss


#%% Test Model
def test(model, test_loader):
    model.eval()
    for batch_idx, (data,target,id,muscle_mua_change) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        if batch_idx == 0:
            id_used = id
            all_output = 100*output
            all_target = 100*target
            error = 100*(output - target)
            muscle_mua_chage_used = 100*muscle_mua_change
        else:
            id_used = np.concatenate((id_used,id))
            all_output = np.concatenate((all_output, 100*output))
            all_target = np.concatenate((all_target, 100*target))
            error = np.concatenate((error, 100*(output - target)))
            muscle_mua_chage_used = np.concatenate((muscle_mua_chage_used, 100*muscle_mua_change))
        if batch_idx % int(0.1*len(test_loader)) == 0:
            print(f"[test] batch:{batch_idx}/{len(test_loader)}({100*batch_idx/len(test_loader):.2f}%)")
    
    df = pd.DataFrame({'id': list(id_used),
                       'output_ijv_SO2' : list(all_output[:,0]),
                       'output_muscle_SO2' : list(all_output[:,1]),
                       'target_ijv_SO2' : list(all_target[:,0]),
                       'target_muscle_SO2' : list(all_target[:,1]),
                       'error_ijv_SO2' : list(error[:,0]),
                       'error_muscle_SO2' : list(error[:,0]),
                       'muscle_mua_change' : list(np.round(10*muscle_mua_chage_used)/10)})
    
    return df


if __name__ == "__main__":
    train_num = 500
    test_num = 50
    #%%
    BATCH_SIZE = 128
    EPOCH = 1000
    lr = 0.0001
    result_folder = "PredictionModel_train_yourself_2_output_large"
    os.makedirs(os.path.join("model_save", result_folder), exist_ok=True)
    
    test_folder = os.path.join("dataset", "prediction_result", "test")
    test_dataset = myDataset(test_folder, test_num, bloodConc, len(bloodConc), len(test_SO2), len(muscle_SO2))
    print(f'test dataset size : {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    torch.save(test_loader, os.path.join("model_save", result_folder, 'test_loader.pth'))
    
    # load model
    model = PredictionModel().cuda()
    # model.load_state_dict(torch.load(os.path.join("prediction_model", "prediction_model.pth")))

    # train
    train_folder = os.path.join("dataset", "prediction_result", "train")
    train_dataset = myDataset(train_folder, train_num, bloodConc, len(bloodConc), len(test_SO2), len(muscle_SO2))
    print(f'train dataset size : {len(train_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # train_0
    # train_dataset_temp = []
    # for x,y,id,muscle_SO2 in train_dataset:
    #     # muscle_SO2 = int(muscle_SO2*1000)
    #     # print(muscle_SO2)
    #     if int(muscle_SO2*1000) in [-5, -1, 0, 1, 5]:
    #         train_dataset_temp.append((x,y,id,muscle_SO2))
    # # train_folder = os.path.join("dataset", "prediction_result", "train")
    # # train_dataset = myDataset(train_folder, train_num, bloodConc, len(bloodConc), len(train_SO2))
    # print(f'train dataset size : {len(train_dataset_temp)}')
    # train_loader = DataLoader(train_dataset_temp, batch_size=BATCH_SIZE, shuffle=True)
    # print(f'train loader size : {len(train_loader)}')
    

    # # train model
    start_time = time.time()
    model = PredictionModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    trlog = train(model, optimizer, criterion, train_loader, train_loader, EPOCH, BATCH_SIZE, lr)
    end_time = time.time()
    print(f'elapsed time : {end_time-start_time:.3f} sec')
    trlog['elapsed_time'] = end_time-start_time
    trlog['train_size'] = len(train_dataset)
    trlog['test_size'] = len(test_dataset)

    # save result 
    with open(os.path.join("model_save", result_folder, "trlog.json"), 'w') as f:
        json.dump(trlog, f, indent=4)  
    torch.save(test_loader, os.path.join("model_save", result_folder, 'test_loader.pth'))
    
    # load result
    with open(os.path.join("model_save", result_folder, "trlog.json"), 'r') as f:
        trlog = json.load(f)
    
    # load model
    model = PredictionModel().cuda()
    model.load_state_dict(torch.load(trlog['best_model']))

    # test model
    df = test(model, train_loader)  
    df.to_csv(os.path.join("model_save", result_folder, "test.csv"), index=False)
            
            
        
        