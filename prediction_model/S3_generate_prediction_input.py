import os
import json
import pandas as pd
import numpy as np
import sys
os.chdir(sys.path[0])

#%%
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
#%%
def gen_prediction_input(num : int, train_or_test: str, SO2_used : list):
    for id in range(num):
        for blc in bloodConc:
            prediction_input = np.empty((len(SO2_used),4*len(wavelength)+3)) # T1_large_SDS1/SDS2 T1_small_SDS1/SDS2 T2_large_SDS1/SDS2 T2_small_SDS1/SDS2 bloodConc ans id
            for i, s in enumerate(SO2_used):
                surrogate_result_T1 = pd.read_csv(os.path.join("dataset", "surrogate_result", train_or_test, 
                                                            f'bloodConc_{blc}', 'SO2_0.7', f'{id}_{train_or_test}.csv'))
                surrogate_result_T2 = pd.read_csv(os.path.join("dataset", "surrogate_result", train_or_test, 
                                                            f'bloodConc_{blc}', f'SO2_{s}', f'{id}_{train_or_test}.csv'))
                prediction_input[i][:20] = surrogate_result_T1['largeIJV_SDS1'] / surrogate_result_T1['largeIJV_SDS2']
                prediction_input[i][20:40] = surrogate_result_T1['smallIJV_SDS1'] / surrogate_result_T1['smallIJV_SDS2']     
                prediction_input[i][40:60] = surrogate_result_T2['largeIJV_SDS1'] / surrogate_result_T2['largeIJV_SDS2']
                prediction_input[i][60:80] = surrogate_result_T2['smallIJV_SDS1'] / surrogate_result_T2['smallIJV_SDS2']
                prediction_input[i][80] = blc
                prediction_input[i][81] = s-0.7 # answer
                prediction_input[i][82] = id # for analyzing used
            np.save(os.path.join("dataset", "prediction_result", train_or_test, f"{id}_blc_{blc}.npy"), prediction_input)

if __name__ == "__main__":
    os.makedirs(os.path.join("dataset", "prediction_result", "train"), exist_ok=True)
    os.makedirs(os.path.join("dataset", "prediction_result", "test"), exist_ok=True)
    gen_prediction_input(30, 'train', train_SO2)
    gen_prediction_input(3, 'test', test_SO2)
    
    