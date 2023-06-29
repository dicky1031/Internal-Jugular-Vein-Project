# %%
import os
import json
import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
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
# def save_prediction_input(prediction_input : pd, start : int, end : int, condition : int):
#     data = []
#     count = 0
#     for i in range(condition):
#         for r in range(start, end):
#             if count == 0:
#                 data = prediction_input[prediction_input['id']==f"{i}_{r}"]
#             else:
#                 data = pd.concat((data, prediction_input[prediction_input['id']==f"{i}_{r}"]))
#             count += 1
#     return data
# %%
muscle_types = ['muscle_0', 'muscle_1', 'muscle_3', 'muscle_5', 'muscle_10', 'muscle_uniform']
mus_types = ['low', 'medium', 'high']
subject = 'ctchen'
# %%
for mus_type in mus_types:
    for muscle_type in muscle_types:
# def gen_precition_input(mus_type, muscle_type):        
        print(f'Now processing mus_type : {mus_type}, muscle_type : {muscle_type} ...')
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'low_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'medium_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'high_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'all_absorption'), exist_ok=True)

        # %%
        based_ijv_SO2 = 0.7
        based_muscle_SO2 = 0.7
        prediction_input = {}

        for i in range(len(wavelength)):
            prediction_input[f'large_{wavelength[i]}nm'] = []
        for i in range(len(wavelength)):
            prediction_input[f'small_{wavelength[i]}nm'] = []
        # for i in range(len(wavelength)):
        #     prediction_input[f'T2_large_{wavelength[i]}nm'] = []
        # for i in range(len(wavelength)):
        #     prediction_input[f'T2_small_{wavelength[i]}nm'] = []
        prediction_input['blc'] = []
        prediction_input['ijv_SO2_change'] = []
        prediction_input['id'] = []
        prediction_input['mua_rank'] = []
        prediction_input['muscle_SO2_change'] = []

        count = 0
        for wl_idx in range(len(wavelength)):
            dataset_large = pd.read_csv(os.path.join('dataset', subject, f'{subject}_dataset_large_{muscle_type}', f'{mus_type}', f'{wavelength[wl_idx]}nm_mus_{wl_idx+1}.csv'))
            dataset_small = pd.read_csv(os.path.join('dataset', subject, f'{subject}_dataset_small_{muscle_type}', f'{mus_type}', f'{wavelength[wl_idx]}nm_mus_{wl_idx+1}.csv'))
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
                        
                        prediction_input[f'large_{wavelength[wl_idx]}nm'] += list((R_T2_large_SDS1/R_T2_large_SDS2).to_numpy() - (R_T1_large_SDS1/R_T1_large_SDS2).to_numpy())
                        prediction_input[f'small_{wavelength[wl_idx]}nm'] += list((R_T2_small_SDS1/R_T2_small_SDS2).to_numpy() - (R_T1_small_SDS1/R_T1_small_SDS2).to_numpy())
                        # prediction_input[f'T2_large_{wavelength[wl_idx]}nm'] += list(R_T2_large_SDS1/R_T2_large_SDS2)
                        # prediction_input[f'T2_small_{wavelength[wl_idx]}nm'] += list(R_T2_small_SDS1/R_T2_small_SDS2)
                        
                        # print(f'blc : {blc}, used_ijv_SO2 : {used_ijv_SO2}, used_muscle_SO2 : {used_muscle_SO2}, {R_T2.shape}')
        for blc in bloodConc:
            for used_ijv_SO2 in test_SO2:
                for used_muscle_SO2 in muscle_SO2:
                    if abs(used_ijv_SO2-based_ijv_SO2) < abs(used_muscle_SO2-based_muscle_SO2):
                        continue
                    prediction_input['blc'] += [blc]*20
                    prediction_input['ijv_SO2_change'] += [used_ijv_SO2-based_ijv_SO2]*20
                    prediction_input['id'] += [count]*20
                    prediction_input['mua_rank'] += [i for i in range(20)]
                    count += 1
                    prediction_input['muscle_SO2_change'] += [used_muscle_SO2-based_muscle_SO2]*20


        prediction_input = pd.DataFrame(prediction_input)
        prediction_input.to_csv(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'all_absorption', 'prediction_input.csv'), index=False)
        all_prediction_input = prediction_input.to_numpy()
        np.save(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'all_absorption', 'prediction_input.npy'), all_prediction_input)
        condition = count

        # %%
        # data = save_prediction_input(prediction_input, start=0, end=7, condition=condition)
        data = prediction_input[prediction_input['mua_rank']<= 7]
        data.to_csv(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'high_absorption', 'prediction_input.csv'), index=False)
        data = data.to_numpy()
        np.save(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'high_absorption', 'prediction_input.npy'), data)

        # data = save_prediction_input(prediction_input, start=7, end=14, condition=condition)
        data = prediction_input[(prediction_input['mua_rank']>7) & (prediction_input['mua_rank']<=14)]
        data.to_csv(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'medium_absorption', 'prediction_input.csv'), index=False)
        data = data.to_numpy()
        np.save(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'medium_absorption', 'prediction_input.npy'), data)

        # data = save_prediction_input(prediction_input, start=14, end=20, condition=condition)
        data = prediction_input[(prediction_input['mua_rank']>14) & (prediction_input['mua_rank']<=20)]
        data.to_csv(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'low_absorption', 'prediction_input.csv'), index=False)
        data = data.to_numpy()
        np.save(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}', 'low_absorption', 'prediction_input.npy'), data)

# products = []
# for mus_type in mus_types:
#     for muscle_type in muscle_types:
#         products.append((mus_type,muscle_type))

# Parallel(n_jobs=-5)(delayed(gen_precition_input)(mus_type, muscle_type) for mus_type, muscle_type in products)
