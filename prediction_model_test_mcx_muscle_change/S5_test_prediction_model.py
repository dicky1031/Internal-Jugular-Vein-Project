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
    #%% 
    pretrained_model_folder = "prediction_model_formula2"
    BATCH_SIZE = 256
    result_folder = os.path.join("low_scatter_prediction_input_muscle_1", "all_absorption")
    os.makedirs(os.path.join("model_test", result_folder), exist_ok=True)
    subject = 'kb'
    test_folder = os.path.join("dataset", subject, "low_scatter_prediction_input_muscle_1", "all_absorption")
    
    # load pre-trained model
    # load result
    with open(os.path.join("model_save", pretrained_model_folder, "trlog.json"), 'r') as f:
        trlog = json.load(f)
    # load model
    model = PredictionModel().cuda()
    model.load_state_dict(torch.load(trlog['best_model']))
    
    # test loader 
    test_dataset = myDataset(folder=test_folder)
    print(f'test dataset size : {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    torch.save(test_loader, os.path.join("model_test", result_folder, 'test_loader.pth'))
    
    # test model
    df = test(model, test_loader)  
    df.to_csv(os.path.join("model_test", result_folder, "test.csv"), index=False)
            
            
        
        