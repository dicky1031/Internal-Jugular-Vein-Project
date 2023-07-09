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
def gen_prediction_input(num : int, train_or_test: str, SO2_used : list, outputpath : str):
    for id in range(num):
        print(f'now processing {train_or_test}_{id}...')
        for blc in bloodConc:
            prediction_input = np.empty((len(SO2_used),len(wavelength)*len(wavelength)+5)) # T1_large_SDS1/SDS2 T1_small_SDS1/SDS2 T2_large_SDS1/SDS2 T2_small_SDS1/SDS2 bloodConc ans id
            for i, s in enumerate(SO2_used):
                surrogate_result_T1 = pd.read_csv(os.path.join("dataset", "surrogate_result", train_or_test, 
                                                            f'bloodConc_{blc}', 'SO2_0.7', f'{id}_{train_or_test}.csv'))
                surrogate_result_T2 = pd.read_csv(os.path.join("dataset", "surrogate_result", train_or_test, 
                                                            f'bloodConc_{blc}', f'SO2_{s}', f'{id}_{train_or_test}.csv'))
                for wl_idx in range(len(wavelength)):
                    prediction_input[i][wl_idx*20 : wl_idx*20+20] = (surrogate_result_T2['largeIJV_SDS1'] / surrogate_result_T2['largeIJV_SDS2'][wl_idx]) / (surrogate_result_T2['smallIJV_SDS1'] / surrogate_result_T2['smallIJV_SDS2'][wl_idx]) - (surrogate_result_T1['largeIJV_SDS1'] / surrogate_result_T1['largeIJV_SDS2'][wl_idx]) / (surrogate_result_T1['smallIJV_SDS1'] / surrogate_result_T1['smallIJV_SDS2'][wl_idx])
                    # prediction_input[i][400+wl_idx*20 : 400+wl_idx*20+20] = surrogate_result_T2['smallIJV_SDS1'] / surrogate_result_T2['smallIJV_SDS2'][wl_idx] - surrogate_result_T1['smallIJV_SDS1'] / surrogate_result_T1['smallIJV_SDS2'][wl_idx]     
                # prediction_input[i][40:60] = surrogate_result_T2['largeIJV_SDS1'] / surrogate_result_T2['largeIJV_SDS2']
                # prediction_input[i][60:80] = surrogate_result_T2['smallIJV_SDS1'] / surrogate_result_T2['smallIJV_SDS2']
                prediction_input[i][400] = blc
                prediction_input[i][401] = s-0.7 # answer
                prediction_input[i][402] = id # for analyzing used
                prediction_input[i][403] = -1 # mua_rank
                prediction_input[i][404] = -1 # muscle_SO2
                
            np.save(os.path.join("dataset", outputpath, train_or_test, f"{id}_blc_{blc}.npy"), prediction_input)

if __name__ == "__main__":
    train_num = 10000
    test_num = 200
    outputpath = 'prediction_model_formula4'
    #%%
    os.makedirs(os.path.join("dataset", outputpath, "train"), exist_ok=True)
    os.makedirs(os.path.join("dataset", outputpath, "test"), exist_ok=True)
    gen_prediction_input(train_num, 'train', train_SO2, outputpath)
    gen_prediction_input(test_num, 'test', test_SO2, outputpath)
    
    