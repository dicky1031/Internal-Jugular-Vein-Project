import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os
import sys
import json
os.chdir(sys.path[0])
from Preprocessing import dataload, data_preprocess
from surrogate_model import ANN

#%% train model
def train(model, optimizer, criterion, train_loader, epoch, batch_size, lr):
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
        for batch_idx, (data,target) in enumerate(train_loader):
            data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % int(0.1*len(train_loader)) == 0:
                print(f"[train] ep:{ep}/{epoch}({100*ep/epoch:.2f}%) batch:{batch_idx}/{len(train_loader)}({100*batch_idx/len(train_loader):.2f}%)\
                      loss={tr_loss/(batch_idx+1)}")
        trlog['train_loss'].append(tr_loss/len(train_loader))
        min_loss = test(trlog,ep,min_loss)
        
    
    return trlog

def test(trlog,ep,min_loss):
    model.eval()
    ts_loss = 0
    for batch_idx, (data,target) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        ts_loss += loss.item()
        
    print(f"[test] batch:{batch_idx}/{len(test_loader)}({100*batch_idx/len(test_loader):.2f}%) loss={ts_loss/len(test_loader)}")
    trlog['test_loss'].append(ts_loss/len(test_loader))
    
    if min_loss > ts_loss/len(test_loader):
        min_loss = ts_loss/len(test_loader)
        trlog['best_model'] = os.path.join("model_save",result_folder,f"ep_{ep}_loss_{min_loss}.pth")
        torch.save(model.state_dict(), os.path.join("model_save",result_folder,f"ep_{ep}_loss_{min_loss}.pth"))
            
    return min_loss
    
#%%   
if __name__ == "__main__":
    subject = 'ctchen'
    result_folder = f"{subject}_large"
    dataset = f"{subject}_large_dataset.npy"
    epoch = 2
    batch_size = 128
    test_split = 0.2
    lr = 0.001
    SDS1 = 2 
    SDS2 = 16
    
    #%% Run Training
    os.makedirs(os.path.join("model_save", result_folder), exist_ok=True)
    root = os.path.join("dataset",dataset)
    mus_set_path = os.path.join("OPs_used", "mus_set.npy")
    mua_set_path = os.path.join("OPs_used", "mua_set.npy")
    # need shuffle or not
    shuffle_dataset = True
    # random seed of shuffle 
    random_seed = 703
    dataset = dataload(root,mus_set_path,mua_set_path, SDS1, SDS2)
    train_loader, test_loader = data_preprocess(dataset, batch_size, test_split, shuffle_dataset, random_seed)
    # train model
    model = ANN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    trlog = train(model, optimizer, criterion, train_loader, epoch, batch_size, lr)
    with open(os.path.join("model_save", result_folder, "trlog.json"), 'w') as f:
        json.dump(trlog, f, indent=4)
    torch.save(train_loader, os.path.join("model_save", result_folder, "train_loader.pth"))
    torch.save(test_loader, os.path.join("model_save", result_folder, "test_loader.pth"))
    
    
        
    
    
    
    
    