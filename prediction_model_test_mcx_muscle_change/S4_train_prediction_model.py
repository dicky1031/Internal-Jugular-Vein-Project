import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
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
                       'error_muscle_SO2' : list(error[:,1]),
                       'muscle_mua_change' : list(np.round(10*muscle_mua_chage_used)/10)})
    
    return df


if __name__ == "__main__":
    BATCH_SIZE = 256
    EPOCH = 500
    lr = 0.0005
    result_folder = 'prediction_model2_formula2'
    os.makedirs(os.path.join("model_save", result_folder), exist_ok=True)
    subject = 'kb'
    train_folder_low = os.path.join("dataset", subject, "low_scatter_prediction_input_muscle_0_train", "all_absorption")
    train_folder_medium = os.path.join("dataset", subject, "medium_scatter_prediction_input_muscle_0_train", "all_absorption")
    train_folder_high = os.path.join("dataset", subject, "high_scatter_prediction_input_muscle_0_train", "all_absorption")
    
    test_folder_low = os.path.join("dataset", subject, "low_scatter_prediction_input_muscle_0_test", "all_absorption")
    test_folder_medium = os.path.join("dataset", subject, "medium_scatter_prediction_input_muscle_0_test", "all_absorption")
    test_folder_high = os.path.join("dataset", subject, "high_scatter_prediction_input_muscle_0_test", "all_absorption")
    
    # train loader 
    train_dataset_low = myDataset(train_folder_low)
    train_dataset_medium = myDataset(train_folder_medium)
    train_dataset_high = myDataset(train_folder_high)
    train_dev_sets = ConcatDataset([train_dataset_low, train_dataset_medium, train_dataset_high])
    print(f'train dataset size : {len(train_dev_sets)}')
    train_loader = DataLoader(train_dev_sets, batch_size=BATCH_SIZE, shuffle=True)
    
    # test loader 
    test_dataset_low = myDataset(test_folder_low)
    test_dataset_medium = myDataset(test_folder_medium)
    test_dataset_high = myDataset(test_folder_high)
    test_dev_sets = ConcatDataset([test_dataset_low, test_dataset_medium, test_dataset_high])
    print(f'test dataset size : {len(test_dev_sets)}')
    test_loader = DataLoader(test_dev_sets, batch_size=BATCH_SIZE, shuffle=False)
    torch.save(test_loader, os.path.join("model_save", result_folder, 'test_loader.pth'))
    
    # # train model
    start_time = time.time()
    model = PredictionModel2().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    trlog = train(model, optimizer, criterion, train_loader, train_loader, EPOCH, BATCH_SIZE, lr)
    end_time = time.time()
    print(f'elapsed time : {end_time-start_time:.3f} sec')
    trlog['elapsed_time'] = end_time-start_time
    trlog['train_size'] = len(train_dev_sets)
    trlog['test_size'] = len(test_dev_sets)

    # save result 
    with open(os.path.join("model_save", result_folder, "trlog.json"), 'w') as f:
        json.dump(trlog, f, indent=4)  
    torch.save(test_loader, os.path.join("model_save", result_folder, 'test_loader.pth'))
    
    # test model
    df = test(model, test_loader)  
    df.to_csv(os.path.join("model_save", result_folder, "test.csv"), index=False)
        