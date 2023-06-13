# %%
import os
import json
import pandas as pd
import numpy as np
import sys

# %%
os.makedirs(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'low_absorption'), exist_ok=True)
os.makedirs(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'medium_absorption'), exist_ok=True)
os.makedirs(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'high_absorption'), exist_ok=True)
os.makedirs(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'all_absorption'), exist_ok=True)

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
based_ijv_SO2 = 0.7
based_muscle_SO2 = 0.7
prediction_input = {}

for i in range(len(wavelength)):
    prediction_input[f'T1_large_{wavelength[i]}nm'] = []
for i in range(len(wavelength)):
    prediction_input[f'T1_small_{wavelength[i]}nm'] = []
for i in range(len(wavelength)):
    prediction_input[f'T2_large_{wavelength[i]}nm'] = []
for i in range(len(wavelength)):
    prediction_input[f'T2_small_{wavelength[i]}nm'] = []
prediction_input['blc'] = []
prediction_input['ijv_SO2_change'] = []
prediction_input['id'] = []
prediction_input['muscle_SO2_change'] = []

count = 0
for wl_idx in range(len(wavelength)):
    dataset_large = pd.read_csv(os.path.join('dataset', 'kb', 'kb_dataset_large', 'low', f'{wavelength[wl_idx]}nm_mus_{wl_idx+1}.csv'))
    dataset_small = pd.read_csv(os.path.join('dataset', 'kb', 'kb_dataset_small', 'low', f'{wavelength[wl_idx]}nm_mus_{wl_idx+1}.csv'))
    for blc in bloodConc:
        for used_ijv_SO2 in test_SO2:
            for used_muscle_SO2 in muscle_SO2:
                if abs(used_ijv_SO2-based_ijv_SO2) < abs(used_muscle_SO2-based_muscle_SO2):
                    continue
                R_T1_large = dataset_large[(dataset_large['bloodConc']==blc) & (dataset_large['used_SO2']==based_ijv_SO2) & (dataset_large['muscle_SO2']==based_muscle_SO2)]
                R_T1_large_SDS1 = R_T1_large['SDS_1']
                R_T1_large_SDS2 = R_T1_large['SDS_11']
                
                R_T1_small = dataset_small[(dataset_small['bloodConc']==blc) & (dataset_small['used_SO2']==based_ijv_SO2) & (dataset_small['muscle_SO2']==based_muscle_SO2)]
                R_T1_small_SDS1 = R_T1_small['SDS_1']
                R_T1_small_SDS2 = R_T1_small['SDS_11']
                
                R_T2_large = dataset_large[(dataset_large['bloodConc']==blc) & (dataset_large['used_SO2']==used_ijv_SO2) & (dataset_large['muscle_SO2']==used_muscle_SO2)]
                R_T2_large_SDS1 = R_T2_large['SDS_1']
                R_T2_large_SDS2 = R_T2_large['SDS_11']
                
                R_T2_small = dataset_small[(dataset_small['bloodConc']==blc) & (dataset_small['used_SO2']==used_ijv_SO2) & (dataset_small['muscle_SO2']==used_muscle_SO2)]
                R_T2_small_SDS1 = R_T2_small['SDS_1']
                R_T2_small_SDS2 = R_T2_small['SDS_11']
                
                prediction_input[f'T1_large_{wavelength[wl_idx]}nm'] += list(R_T1_large_SDS1/R_T1_large_SDS2)
                prediction_input[f'T1_small_{wavelength[wl_idx]}nm'] += list(R_T1_small_SDS1/R_T1_small_SDS2)
                prediction_input[f'T2_large_{wavelength[wl_idx]}nm'] += list(R_T2_large_SDS1/R_T2_large_SDS2)
                prediction_input[f'T2_small_{wavelength[wl_idx]}nm'] += list(R_T2_small_SDS1/R_T2_small_SDS2)
                
                # print(f'blc : {blc}, used_ijv_SO2 : {used_ijv_SO2}, used_muscle_SO2 : {used_muscle_SO2}, {R_T2.shape}')
for blc in bloodConc:
    for used_ijv_SO2 in test_SO2:
        for used_muscle_SO2 in muscle_SO2:
            if abs(used_ijv_SO2-based_ijv_SO2) < abs(used_muscle_SO2-based_muscle_SO2):
                continue
            prediction_input['blc'] += [blc]*20
            prediction_input['ijv_SO2_change'] += [used_ijv_SO2]*20
            prediction_input['id'] += [f'{count}_{i}' for i in range(20)]
            count += 1
            prediction_input['muscle_SO2_change'] += [used_muscle_SO2]*20


prediction_input = pd.DataFrame(prediction_input)
prediction_input.to_csv(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'all_absorption', 'prediction_input.csv'), index=False)
all_prediction_input = prediction_input.to_numpy()
np.save(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'all_absorption', 'prediction_input.npy'), all_prediction_input)
condition = 10580

# %%
def save_prediction_input(prediction_input : pd, start : int, end : int):
    data = []
    count = 0
    for i in range(condition):
        for r in range(start, end):
            if count == 0:
                data = prediction_input[prediction_input['id']==f"{i}_{r}"]
            else:
                data = pd.concat((data, prediction_input[prediction_input['id']==f"{i}_{r}"]))
            count += 1
    return data

# %%
data = save_prediction_input(prediction_input, start=0, end=7)
data.to_csv(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'high_absorption', 'prediction_input.csv'), index=False)
data = data.to_numpy()
np.save(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'high_absorption', 'prediction_input.npy'), data)

data = save_prediction_input(prediction_input, start=7, end=14)
data.to_csv(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'medium_absorption', 'prediction_input.csv'), index=False)
data = data.to_numpy()
np.save(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'medium_absorption', 'prediction_input.npy'), data)

data = save_prediction_input(prediction_input, start=14, end=20)
data.to_csv(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'low_absorption', 'prediction_input.csv'), index=False)
data = data.to_numpy()
np.save(os.path.join('dataset', 'kb', 'low_scatter_prediction_input', 'low_absorption', 'prediction_input.npy'), data)


