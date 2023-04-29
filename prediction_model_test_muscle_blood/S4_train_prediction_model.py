import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ANN_models import PredictionModel
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

#%% Test Model
def test(model, test_loader):
    model.eval()
    for batch_idx, (data,target,id,muscle_mua_change) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output.detach().cpu().numpy().reshape(-1,1)
        target = target.detach().cpu().numpy().reshape(-1,1)
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
        
        print(f"[test] batch:{batch_idx}/{len(test_loader)}({100*batch_idx/len(test_loader):.2f}%)")
    
    df = pd.DataFrame({'id': list(id_used),
                       'output' : list(all_output.reshape(-1)),
                       'target' : list(all_target.reshape(-1)),
                       'error' : list(error.reshape(-1)),
                       'muscle_mua_change' : list(np.round(muscle_mua_chage_used))})
    
    return df


if __name__ == "__main__":
    train_num = 20
    test_num = 3
    #%%
    BATCH_SIZE = 256
    result_folder = "result1"
    os.makedirs(os.path.join("model_save", result_folder), exist_ok=True)
    
    trest_folder = os.path.join("dataset", "prediction_result", "test")
    test_dataset = myDataset(trest_folder, test_num, bloodConc, len(bloodConc), len(test_SO2), len(muscle_SO2))
    print(f'test dataset size : {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    torch.save(test_loader, os.path.join("model_save", result_folder, 'test_loader.pth'))
    
    # load model
    model = PredictionModel().cuda()
    model.load_state_dict(torch.load(os.path.join("prediction_model", "prediction_model.pth")))
    
    # test model
    df = test(model, test_loader)  
    df.to_csv(os.path.join("model_save", result_folder, "test.csv"), index=False)
            
            
        
        