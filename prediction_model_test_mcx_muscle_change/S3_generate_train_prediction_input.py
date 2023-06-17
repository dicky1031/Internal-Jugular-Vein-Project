# %%
import os
import json
import pandas as pd
import numpy as np
import sys

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
muscle_types = ['muscle_0']
mus_types = ['low', 'medium', 'high']
subject = 'ctchen'
# %%
for mus_type in mus_types:
    for muscle_type in muscle_types:
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_train', 'low_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_train', 'medium_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_train', 'high_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_train', 'all_absorption'), exist_ok=True)
        
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_test', 'low_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_test', 'medium_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_test', 'high_absorption'), exist_ok=True)
        os.makedirs(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_test', 'all_absorption'), exist_ok=True)

        

        # %%
        based_ijv_SO2 = 0.7
        based_muscle_SO2 = 0.7
        prediction_input_test = {}
        for i in range(len(wavelength)):
            prediction_input_test[f'large_{wavelength[i]}nm'] = []
        for i in range(len(wavelength)):
            prediction_input_test[f'small_{wavelength[i]}nm'] = []
        # for i in range(len(wavelength)):
        #     prediction_input[f'T2_large_{wavelength[i]}nm'] = []
        # for i in range(len(wavelength)):
        #     prediction_input[f'T2_small_{wavelength[i]}nm'] = []
        prediction_input_test['blc'] = []
        prediction_input_test['ijv_SO2_change'] = []
        prediction_input_test['id'] = []
        prediction_input_test['muscle_SO2_change'] = []
        
        prediction_input_train = {}
        for i in range(len(wavelength)):
            prediction_input_train[f'large_{wavelength[i]}nm'] = []
        for i in range(len(wavelength)):
            prediction_input_train[f'small_{wavelength[i]}nm'] = []
        # for i in range(len(wavelength)):
        #     prediction_input[f'T2_large_{wavelength[i]}nm'] = []
        # for i in range(len(wavelength)):
        #     prediction_input[f'T2_small_{wavelength[i]}nm'] = []
        prediction_input_train['blc'] = []
        prediction_input_train['ijv_SO2_change'] = []
        prediction_input_train['id'] = []
        prediction_input_train['muscle_SO2_change'] = []
        

        count = 0
        for wl_idx in range(len(wavelength)):
            dataset_large = pd.read_csv(os.path.join('dataset', subject, f'kb_dataset_large_{muscle_type}', f'{mus_type}', f'{wavelength[wl_idx]}nm_mus_{wl_idx+1}.csv'))
            dataset_small = pd.read_csv(os.path.join('dataset', subject, f'kb_dataset_small_{muscle_type}', f'{mus_type}', f'{wavelength[wl_idx]}nm_mus_{wl_idx+1}.csv'))
            for blc in bloodConc:
                for used_ijv_SO2 in test_SO2:
                    R_T1_large = dataset_large[(dataset_large['bloodConc']==blc) & (dataset_large['used_SO2']==based_ijv_SO2) & (dataset_large['muscle_SO2']==based_muscle_SO2)]
                    R_T1_large_SDS1 = R_T1_large['SDS_1']
                    R_T1_large_SDS2 = R_T1_large['SDS_11']
                    
                    R_T1_small = dataset_small[(dataset_small['bloodConc']==blc) & (dataset_small['used_SO2']==based_ijv_SO2) & (dataset_small['muscle_SO2']==based_muscle_SO2)]
                    R_T1_small_SDS1 = R_T1_small['SDS_1']
                    R_T1_small_SDS2 = R_T1_small['SDS_11']
                    
                    R_T2_large = dataset_large[(dataset_large['bloodConc']==blc) & (dataset_large['used_SO2']==used_ijv_SO2) & (dataset_large['muscle_SO2']==based_muscle_SO2)]
                    R_T2_large_SDS1 = R_T2_large['SDS_1']
                    R_T2_large_SDS2 = R_T2_large['SDS_11']
                    
                    R_T2_small = dataset_small[(dataset_small['bloodConc']==blc) & (dataset_small['used_SO2']==used_ijv_SO2) & (dataset_small['muscle_SO2']==based_muscle_SO2)]
                    R_T2_small_SDS1 = R_T2_small['SDS_1']
                    R_T2_small_SDS2 = R_T2_small['SDS_11']
                    
                    temp = list((R_T2_large_SDS1/R_T2_large_SDS2).to_numpy() - (R_T1_large_SDS1/R_T1_large_SDS2).to_numpy())
                    prediction_input_test[f'large_{wavelength[wl_idx]}nm'] += temp[0::2]
                    prediction_input_train[f'large_{wavelength[wl_idx]}nm'] += temp[1::2]
                    
                    temp = list((R_T2_small_SDS1/R_T2_small_SDS2).to_numpy() - (R_T1_small_SDS1/R_T1_small_SDS2).to_numpy())
                    prediction_input_test[f'small_{wavelength[wl_idx]}nm'] += temp[0::2]
                    prediction_input_train[f'small_{wavelength[wl_idx]}nm'] += temp[1::2]
                    # prediction_input[f'T2_large_{wavelength[wl_idx]}nm'] += list(R_T2_large_SDS1/R_T2_large_SDS2)
                    # prediction_input[f'T2_small_{wavelength[wl_idx]}nm'] += list(R_T2_small_SDS1/R_T2_small_SDS2)
                    
                    # print(f'blc : {blc}, used_ijv_SO2 : {used_ijv_SO2}, used_muscle_SO2 : {used_muscle_SO2}, {R_T2.shape}')
        for blc in bloodConc:
            for used_ijv_SO2 in test_SO2:
                prediction_input_test['blc'] += [blc]*10
                prediction_input_test['ijv_SO2_change'] += [used_ijv_SO2-based_ijv_SO2]*10
                prediction_input_test['id'] += [f'{count}_{i}' for i in range(10)]
                prediction_input_test['muscle_SO2_change'] += [0.0]*10
                
                prediction_input_train['blc'] += [blc]*10
                prediction_input_train['ijv_SO2_change'] += [used_ijv_SO2-based_ijv_SO2]*10
                prediction_input_train['id'] += [f'{count}_{i}' for i in range(10)]
                prediction_input_train['muscle_SO2_change'] += [0.0]*10
                count += 1

        if mus_type == 'low':
            prediction_input_test = pd.DataFrame(prediction_input_test)
            prediction_input_test.to_csv(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_test', 'all_absorption', 'prediction_input_test.csv'), index=False)
            all_prediction_input_test = prediction_input_test.to_numpy()
            np.save(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_test', 'all_absorption', 'prediction_input_test.npy'), all_prediction_input_test)
        
        else:
            prediction_input_train = pd.DataFrame(prediction_input_train)
            prediction_input_train.to_csv(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_train', 'all_absorption', 'prediction_input_train.csv'), index=False)
            all_prediction_input_train = prediction_input_train.to_numpy()
            np.save(os.path.join('dataset', subject, f'{mus_type}_scatter_prediction_input_{muscle_type}_train', 'all_absorption', 'prediction_input_train.npy'), all_prediction_input_train)

