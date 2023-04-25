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
def gen_prediction_input(num : int, train_or_test: str):
    for id in range(num):
        for blc in bloodConc:
            prediction_input = np.empty((len(train_SO2),4*len(wavelength)+2)) # T1_large_SDS1/SDS2 T1_small_SDS1/SDS2 T2_large_SDS1/SDS2 T2_small_SDS1/SDS2 bloodConc ANS
            for i, s in enumerate(train_SO2):
                surrogate_result_T1 = pd.read_csv(os.path.join("dataset", "surrogate_result", "train", 
                                                            f'bloodConc_{blc}', 'SO2_0.7', f'{id}.csv'))
                surrogate_result_T2 = pd.read_csv(os.path.join("dataset", "surrogate_result", "train", 
                                                            f'bloodConc_{blc}', f'SO2_{s}', f'{id}.csv'))
                prediction_input[i][:20] = surrogate_result_T1['largeIJV_SDS1'] / surrogate_result_T1['largeIJV_SDS2']
                prediction_input[i][20:40] = surrogate_result_T1['smallIJV_SDS1'] / surrogate_result_T1['smallIJV_SDS2']     
                prediction_input[i][40:60] = surrogate_result_T2['largeIJV_SDS1'] / surrogate_result_T2['largeIJV_SDS2']
                prediction_input[i][60:80] = surrogate_result_T2['smallIJV_SDS1'] / surrogate_result_T2['smallIJV_SDS2']
                prediction_input[i][80] = blc
                prediction_input[i][81] = s-0.7 # answer
            np.save(os.path.join("dataset", train_or_test, "prediction_result", f"{id}_blc_{blc}.npy"), prediction_input)

if __name__ == "__main__":
    os.makedirs(os.path.join("dataset","train", "prediction_result"), exist_ok=True)
    os.makedirs(os.path.join("dataset","test", "prediction_result"), exist_ok=True)
    