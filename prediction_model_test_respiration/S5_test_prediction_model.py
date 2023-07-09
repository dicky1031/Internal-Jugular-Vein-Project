import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ANN_models import PredictionModel, PredictionModel2, PredictionModel3
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
    for batch_idx, (data, target, id, mua_rank) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        if batch_idx == 0:
            id_used = id
            mua_rank_used = mua_rank
            all_output = 100*output
            all_target = 100*target
            error = 100*(output - target)
        else:
            id_used = np.concatenate((id_used,id))
            mua_rank_used = np.concatenate((mua_rank_used, mua_rank))
            all_output = np.concatenate((all_output, 100*output))
            all_target = np.concatenate((all_target, 100*target))
            error = np.concatenate((error, 100*(output - target)))
        if batch_idx % int(0.1*len(test_loader)) == 0:
            print(f"[test] batch:{batch_idx}/{len(test_loader)}({100*batch_idx/len(test_loader):.2f}%)")
    
    df = pd.DataFrame({'id': list(id_used),
                       'mua_rank': list(mua_rank_used),
                       'output_ijv_SO2' : list(all_output[:,0]),
                       'target_ijv_SO2' : list(all_target[:,0]),
                       'error_ijv_SO2' : list(error[:,0])})
    
    return df


if __name__ == "__main__":
    #%% load pre-trained model
    pretrained_model_folder = "prediction_model_formula3_train_on_mcx_again"
    BATCH_SIZE = 512
    subject = 'ctchen'
    result = 'surrogate_formula3_train_on_mcx_again'
    
    # load result
    with open(os.path.join("model_save", subject, pretrained_model_folder, "trlog.json"), 'r') as f:
        trlog = json.load(f)
    # load model
    model = PredictionModel3().cuda()
    model.load_state_dict(torch.load(trlog['best_model']))
    
    # mus_types = ['low', 'high', 'medium']
    mus_types = ['low']
    mua_types = ['all', 'low', 'high', 'medium']
    # ijv_depth = ['+1mm', '+0.5mm', '-0.5mm', '-1mm', 'standard']
    ijv_depth = ['standard']
    ijv_size = ['-50%', '-30%', '-20%', '-10%', 'standard']
    # ijv_size = ['standard']
    
    for using_depth in ijv_depth:
        for using_size in ijv_size:
            for mus_type in mus_types:
                for mua_type in mua_types:
                    result_folder = os.path.join(f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f"{mus_type}_scatter_prediction_input", f"{mua_type}_absorption")
                    
                    os.makedirs(os.path.join("model_test", subject, result, result_folder), exist_ok=True)
                    test_folder = os.path.join("dataset", subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f"{mus_type}_scatter_prediction_input", f"{mua_type}_absorption")
                    
                    # test loader 
                    test_dataset = myDataset(folder=test_folder)
                    print(f'Now processing mus_type : {mus_type}, ijv_depth : {using_depth}, ijv_size : {using_size}')
                    print(f'test dataset size : {len(test_dataset)}')
                    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    torch.save(test_loader, os.path.join("model_test", subject, result, result_folder, 'test_loader.pth'))
                    
                    # test model
                    df = test(model, test_loader)  
                    df.to_csv(os.path.join("model_test", subject, result, result_folder, "test.csv"), index=False)
            
            
        
        