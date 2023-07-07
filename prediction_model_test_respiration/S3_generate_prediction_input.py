# %%
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
def gen_precition_input(wl2_idx, wl_idx, all_dataset_large, all_dataset_small, based_ijv_SO2):
    SDS1_dataset_large = all_dataset_large[all_dataset_large['wavelength']==wavelength[wl_idx]]
    SDS1_dataset_small = all_dataset_small[all_dataset_small['wavelength']==wavelength[wl_idx]]
    SDS2_dataset_large = all_dataset_large[all_dataset_large['wavelength']==wavelength[wl2_idx]]
    SDS2_dataset_small = all_dataset_small[all_dataset_small['wavelength']==wavelength[wl2_idx]]
    output_large = []
    output_small = []
    for blc in bloodConc:
        for used_ijv_SO2 in test_SO2:
            R_T1_large_SDS1 = SDS1_dataset_large[(SDS1_dataset_large['bloodConc']==blc) & (SDS1_dataset_large['used_SO2']==based_ijv_SO2)]['SDS_1'].to_numpy()
            R_T1_large_SDS2 = SDS2_dataset_large[(SDS2_dataset_large['bloodConc']==blc) & (SDS2_dataset_large['used_SO2']==based_ijv_SO2)]['SDS_11'].to_numpy()
            
            R_T1_small_SDS1 = SDS1_dataset_small[(SDS1_dataset_small['bloodConc']==blc) & (SDS1_dataset_small['used_SO2']==based_ijv_SO2)]['SDS_1'].to_numpy()
            R_T1_small_SDS2 = SDS2_dataset_small[(SDS2_dataset_small['bloodConc']==blc) & (SDS2_dataset_small['used_SO2']==based_ijv_SO2)]['SDS_11'].to_numpy()
            
            R_T2_large_SDS1 = SDS1_dataset_large[(SDS1_dataset_large['bloodConc']==blc) & (SDS1_dataset_large['used_SO2']==used_ijv_SO2)]['SDS_1'].to_numpy()
            R_T2_large_SDS2 = SDS2_dataset_large[(SDS2_dataset_large['bloodConc']==blc) & (SDS2_dataset_large['used_SO2']==used_ijv_SO2)]['SDS_11'].to_numpy()
            
            R_T2_small_SDS1 = SDS1_dataset_small[(SDS1_dataset_small['bloodConc']==blc) & (SDS1_dataset_small['used_SO2']==used_ijv_SO2)]['SDS_1'].to_numpy()
            R_T2_small_SDS2= SDS2_dataset_small[(SDS2_dataset_small['bloodConc']==blc) & (SDS2_dataset_small['used_SO2']==used_ijv_SO2)]['SDS_11'].to_numpy()
            
            output_large += list((R_T2_large_SDS1/R_T2_large_SDS2) - (R_T1_large_SDS1/R_T1_large_SDS2))
            output_small += list((R_T2_small_SDS1/R_T2_small_SDS2)- (R_T1_small_SDS1/R_T1_small_SDS2))
    
    return wl2_idx, wl_idx, output_large, output_small

# %%
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
# ijv_depth = ['+1mm', '+0.5mm', '-0.5mm', '-1mm', 'standard']
ijv_depth = ['standard']
# ijv_size = ['-50%', '-30%', '-20%', '-10%', 'standard']
ijv_size = ['-10%']
# mus_types = ['low', 'medium', 'high']
mus_types = ['low']
subject = 'ctchen'
start_time = time.time()
# %%
for using_depth in ijv_depth:
    for using_size in ijv_size:
        for mus_type in mus_types:       
            print(f'Now processing mus_type : {mus_type}, ijv_depth : {using_depth}, ijv_size : {using_size}')
            os.makedirs(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'low_absorption'), exist_ok=True)
            os.makedirs(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'medium_absorption'), exist_ok=True)
            os.makedirs(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'high_absorption'), exist_ok=True)
            os.makedirs(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'all_absorption'), exist_ok=True)

            all_dataset_large = pd.read_csv(os.path.join('dataset', subject, f'{subject}_dataset_large_ijv_depth_{using_depth}_size_{using_size}', f'{mus_type}', 'large_all.csv'))
            all_dataset_small = pd.read_csv(os.path.join('dataset', subject, f'{subject}_dataset_small_ijv_depth_{using_depth}_size_{using_size}', f'{mus_type}', 'small_all.csv'))
            
            # %%
            based_ijv_SO2 = 0.7
            prediction_input = {}
            for j in range(len(wavelength)):
                for i in range(len(wavelength)):
                    prediction_input[f'large_{wavelength[i]}nm_based_on_{wavelength[j]}'] = []
            for j in range(len(wavelength)):
                for i in range(len(wavelength)):
                    prediction_input[f'small_{wavelength[i]}nm_based_on_{wavelength[j]}'] = []
            # for i in range(len(wavelength)):
            #     prediction_input[f'T2_large_{wavelength[i]}nm'] = []
            # for i in range(len(wavelength)):
            #     prediction_input[f'T2_small_{wavelength[i]}nm'] = []
            prediction_input['blc'] = []
            prediction_input['ijv_SO2_change'] = []
            prediction_input['id'] = []
            prediction_input['mua_rank'] = []

            products = []
            for wl2_idx in range(len(wavelength)):
                for wl_idx in range(len(wavelength)):
                    products.append((wl2_idx,wl_idx))

            res = Parallel(n_jobs=-5)(delayed(gen_precition_input)(wl2_idx, wl_idx, all_dataset_large, all_dataset_small, based_ijv_SO2) for wl2_idx, wl_idx in tqdm(products))
            for wl2_idx, wl_idx, output_large, output_small in res:
                prediction_input[f'large_{wavelength[wl_idx]}nm_based_on_{wavelength[wl2_idx]}'] += output_large
                prediction_input[f'small_{wavelength[wl_idx]}nm_based_on_{wavelength[wl2_idx]}'] += output_small
            
            print(f'finish spectrum = {time.time()-start_time} s')
            start_time = time.time()
            
            count = 0
            for blc in bloodConc:
                for used_ijv_SO2 in test_SO2:
                    prediction_input['blc'] += [blc]*20
                    prediction_input['ijv_SO2_change'] += [used_ijv_SO2-based_ijv_SO2]*20
                    prediction_input['id'] += [count]*20
                    prediction_input['mua_rank'] += [i for i in range(20)]
                    count += 1
            print(f'finish spectrum 2 = {time.time()-start_time} s')
            start_time = time.time()

            prediction_input = pd.DataFrame(prediction_input)
            prediction_input.to_csv(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'all_absorption', 'prediction_input.csv'), index=False)
            all_prediction_input = prediction_input.to_numpy()
            np.save(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'all_absorption', 'prediction_input.npy'), all_prediction_input)
            condition = count
            print(f'finish save all = {time.time()-start_time} s')
            start_time = time.time()
            # %%
            # data = save_prediction_input(prediction_input, start=0, end=7, condition=condition)
            data = prediction_input[prediction_input['mua_rank']<= 7]
            data.to_csv(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'high_absorption', 'prediction_input.csv'), index=False)
            data = data.to_numpy()
            np.save(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'high_absorption', 'prediction_input.npy'), data)
            print(f'finish save high = {time.time()-start_time} s')
            start_time = time.time()
            
            # data = save_prediction_input(prediction_input, start=7, end=14, condition=condition)
            data = prediction_input[(prediction_input['mua_rank']>7) & (prediction_input['mua_rank']<=14)]
            data.to_csv(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'medium_absorption', 'prediction_input.csv'), index=False)
            data = data.to_numpy()
            np.save(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'medium_absorption', 'prediction_input.npy'), data)
            print(f'finish save mid = {time.time()-start_time} s')
            start_time = time.time()
            
            # data = save_prediction_input(prediction_input, start=14, end=20, condition=condition)
            data = prediction_input[(prediction_input['mua_rank']>14) & (prediction_input['mua_rank']<=20)]
            data.to_csv(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'low_absorption', 'prediction_input.csv'), index=False)
            data = data.to_numpy()
            np.save(os.path.join('dataset', subject, f'ijv_depth_{using_depth}', f'ijv_size_{using_size}', f'{mus_type}_scatter_prediction_input', 'low_absorption', 'prediction_input.npy'), data)
            print(f'finish save low = {time.time()-start_time} s')





