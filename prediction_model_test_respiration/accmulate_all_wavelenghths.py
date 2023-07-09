# %%
import os
import json
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
import time 
# %% move to current file path
os.chdir(sys.path[0])

# %%
with open(os.path.join("OPs_used", "bloodConc.json"), "r") as f:
    bloodConc = json.load(f)
    bloodConc = bloodConc['bloodConc']
with open(os.path.join("OPs_used", "wavelength.json"), 'r') as f:
    wavelength = json.load(f)
    wavelength = wavelength['wavelength']
with open(os.path.join("OPs_used", "SO2.json"), 'r') as f:
    SO2 = json.load(f)
    train_SO2 = SO2['train_SO2']
    test_SO2 = SO2['test_SO2']
with open(os.path.join('OPs_used', "muscle_SO2.json"), 'r') as f:
    muscle_SO2 = json.load(f)
    muscle_SO2 = muscle_SO2['SO2']
    
# %%
ijv_depth = ['+1mm', '+0.5mm', '-0.5mm', '-1mm', 'standard']
# ijv_depth = ['standard']
# ijv_size = ['-50%', '-30%', '-20%', '-10%', 'standard']
ijv_size = ['standard']
# mus_types = ['low', 'medium', 'high']
mus_types = ['low']
subject = 'ctchen'    
    
for using_depth in ijv_depth:
    for using_size in ijv_size:
        for mus_type in mus_types: 
            count = 0       
            for wl_idx in range(len(wavelength)):
                dataset_large = pd.read_csv(os.path.join('dataset', subject, f'{subject}_dataset_large_ijv_depth_{using_depth}_size_{using_size}', f'{mus_type}', f'{wavelength[wl_idx]}nm_mus_{wl_idx+1}.csv'))
                dataset_large.insert(0, 'wavelength', wavelength[wl_idx])
                dataset_small = pd.read_csv(os.path.join('dataset', subject, f'{subject}_dataset_small_ijv_depth_{using_depth}_size_{using_size}', f'{mus_type}', f'{wavelength[wl_idx]}nm_mus_{wl_idx+1}.csv'))  
                dataset_small.insert(0, 'wavelength', wavelength[wl_idx])
                if count == 0:
                    all_dataset_large = dataset_large
                    all_dataset_small = dataset_small
                else:
                    all_dataset_large = pd.concat((all_dataset_large, dataset_large))
                    all_dataset_small = pd.concat((all_dataset_small, dataset_small))
                count += 1
            all_dataset_large.to_csv(os.path.join('dataset', subject, f'{subject}_dataset_large_ijv_depth_{using_depth}_size_{using_size}', f'{mus_type}', 'large_all.csv'), index=False)
            all_dataset_small.to_csv(os.path.join('dataset', subject, f'{subject}_dataset_small_ijv_depth_{using_depth}_size_{using_size}', f'{mus_type}', 'small_all.csv'), index=False)