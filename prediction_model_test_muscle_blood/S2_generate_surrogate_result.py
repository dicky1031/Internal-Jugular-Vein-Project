import json
import os
import random
import numpy as np
import pandas as pd
import torch
from ANN_models import SurrogateModel
import sys
os.chdir(sys.path[0])

#%% load all we need file
with open(os.path.join("OPs_used","mus_spectrum.json"), "r") as f:
    mus_spectrum = json.load(f)
with open(os.path.join("OPs_used","mua_spectrum.json"), "r") as f:
    mua_spectrum = json.load(f)
with open(os.path.join("OPs_used", "bloodConc.json"), "r") as f:
    bloodConc = json.load(f)
    bloodConc = bloodConc['bloodConc']
with open(os.path.join("OPs_used", "wavelength.json"), 'r') as f:
    used_wl = json.load(f)
    used_wl = used_wl['wavelength']
with open(os.path.join("OPs_used", "SO2.json"), 'r') as f:
    SO2 = json.load(f)
    train_SO2 = SO2['train_SO2']
    test_SO2 = SO2['test_SO2']
with open(os.path.join('OPs_used', "muscle_SO2.json"), 'r') as f:
    muscle_SO2 = json.load(f)
    muscle_SO2 = muscle_SO2['SO2']
mus_set = pd.read_csv(os.path.join("OPs_used","mus_set.csv")).to_numpy()
mua_set = pd.read_csv(os.path.join("OPs_used","mua_set.csv")).to_numpy()
# load surrogate model
large_ijv_model = SurrogateModel().cuda()
large_ijv_model.load_state_dict(torch.load(os.path.join("surrogate_model","large_ANN_model.pth")))
small_ijv_model = SurrogateModel().cuda()
small_ijv_model.load_state_dict(torch.load(os.path.join("surrogate_model","small_ANN_model.pth")))

# load muscle_with_blodd mua spectrum
with open(os.path.join("OPs_used", "muscle_mua_spectrum.json"), 'r') as f:
    muscle_mua_spectrum = json.load(f)

#%% functions
def preprocess_data(arr, mus_set, mua_set):
    OPs_normalized = torch.from_numpy(arr[:,:10]) 
    max_mus = np.max(mus_set, axis=0)[:5]
    max_mua = np.max(mua_set, axis=0)[:5]
    x_max = torch.from_numpy(np.concatenate((max_mus,max_mua)))
    min_mus = np.min(mus_set, axis=0)[:5]
    min_mua = np.min(mua_set, axis=0)[:5]
    x_min = torch.from_numpy(np.concatenate((min_mus,min_mua)))
    OPs_normalized = (OPs_normalized - x_min) / (x_max - x_min)
    SO2_used = torch.from_numpy(arr[:,10]) # SO2
    bloodConc_used = torch.from_numpy(arr[:,11]) # bloodConc
    
    return OPs_normalized, SO2_used, bloodConc_used

def gen_surrogate_result(bloodConc:list, used_SO2:list, mus:dict, mua:dict, muscle_SO2_used:int, train_or_test:str, rangdom_gen:list, id:int):
    for blc in bloodConc:
        for s in used_SO2:
            os.makedirs(os.path.join("dataset", "surrogate_result", train_or_test, f'bloodConc_{blc}', f'muscle_SO2_{muscle_SO2_used}', f'ijv_SO2_{s}'), exist_ok=True)
    for blc in bloodConc:
        surrogate_concurrent_data = {"wavelength" : [f'{wl} nm' for wl in used_wl],
                            "skin_mus": mus["skin"][rangdom_gen[0]],
                            "fat_mus": mus["fat"][rangdom_gen[1]],
                            "muscle_mus": mus["muscle"][rangdom_gen[2]],
                            "ijv_mus": mus["blood"][rangdom_gen[3]],
                            "cca_mus": mus["blood"][rangdom_gen[3]],
                            "skin_mua": mua["skin"][rangdom_gen[5]],
                            "fat_mua": mua["fat"][rangdom_gen[6]]}
        for s in used_SO2:
            surrogate_input = {"wavelength" : [f'{wl} nm' for wl in used_wl],
                            "skin_mus": mus["skin"][rangdom_gen[0]],
                            "fat_mus": mus["fat"][rangdom_gen[1]],
                            "muscle_mus": mus["muscle"][rangdom_gen[2]],
                            "ijv_mus": mus["blood"][rangdom_gen[3]],
                            "cca_mus": mus["blood"][rangdom_gen[3]],
                            "skin_mua": mua["skin"][rangdom_gen[5]],
                            "fat_mua": mua["fat"][rangdom_gen[6]],
                            "muscle_mua" : muscle_mua_spectrum[f'muscle_bloodConc_{blc}_bloodSO2_{muscle_SO2_used}'],
                            "ijv_mua": mua_spectrum[f'ijv_bloodConc_{blc}_bloodSO2_{s}'],
                            "cca_mua": mua["cca"][rangdom_gen[8]],
                            "answer": s,
                            "bloodConc": blc}
            surrogate_input = pd.DataFrame(surrogate_input)
            
            # get surrogate model output
            arr = surrogate_input.to_numpy()
            arr = arr[:,1:].astype(np.float64) # OPs_used
            OPs_normalized, SO2_used, bloodConc_used = preprocess_data(arr, mus_set, mua_set)
            large_reflectance = large_ijv_model(OPs_normalized.to(torch.float32).cuda())
            large_reflectance = torch.exp(-large_reflectance).detach().cpu().numpy()
            small_reflectance = small_ijv_model(OPs_normalized.to(torch.float32).cuda())
            small_reflectance = torch.exp(-small_reflectance).detach().cpu().numpy()
            
            # save reflectance
            surrogate_input['largeIJV_SDS1'] = large_reflectance[:,0]
            surrogate_input['largeIJV_SDS2'] = large_reflectance[:,1]
            surrogate_input['smallIJV_SDS1'] = small_reflectance[:,0]
            surrogate_input['smallIJV_SDS2'] = small_reflectance[:,1]
            
            # save result
            surrogate_input = surrogate_input.drop(columns=['wavelength', 'skin_mus', 'fat_mus', 'muscle_mus', 'ijv_mus', 
                                 'cca_mus', 'skin_mua', 'fat_mua', 'answer', 'bloodConc']) # drop these values for saving memory
            
            surrogate_input.to_csv(os.path.join("dataset", "surrogate_result", train_or_test, f'bloodConc_{blc}', f'muscle_SO2_{muscle_SO2_used}', f'ijv_SO2_{s}', f'{id}_{train_or_test}.csv'), index=False) 
        surrogate_concurrent_data = pd.DataFrame(surrogate_concurrent_data)
        surrogate_concurrent_data.to_csv(os.path.join("dataset", "surrogate_result", train_or_test, f'{id}_{train_or_test}_concurrent.csv'), index=False)
        
        
if __name__ == "__main__":
    test_num = 3
    
    #%num of spectrum used
    total_num = 10

    # get mus spectrum 
    mus = {}
    tissue = ['skin', 'fat', 'muscle', 'blood']
    for t in tissue:
        mus[t] = pd.DataFrame(mus_spectrum[t]).to_numpy()

    # get mua spectrum 
    mua = {}
    tissue = ["skin", "fat", "cca", "muscle"]
    for t in tissue:
        mua[t] = pd.DataFrame(mua_spectrum[t]).to_numpy()
    
    # generate testset
    for id in range(test_num):
        for muscle_SO2_used in muscle_SO2:
            print(f'now processing test_{id} muscle_SO2 {muscle_SO2_used}...')
            rangdom_gen = [2*random.randint(0, total_num-1)+1,2*random.randint(0, total_num-1)+1,2*random.randint(0, total_num-1)+1,
                        2*random.randint(0, total_num-1)+1,2*random.randint(0, total_num-1)+1,2*random.randint(0, total_num-1)+1,
                        2*random.randint(0, total_num-1)+1,2*random.randint(0, total_num-1)+1,2*random.randint(0, total_num-1)+1] # generate odds for choose training input
            gen_surrogate_result(bloodConc, test_SO2, mus, mua, muscle_SO2_used, "test", rangdom_gen, id)
        
