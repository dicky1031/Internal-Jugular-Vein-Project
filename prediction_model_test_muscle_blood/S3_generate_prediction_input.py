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
with open(os.path.join('OPs_used', "muscle_SO2.json"), 'r') as f:
    muscle_SO2 = json.load(f)
    muscle_SO2 = muscle_SO2['SO2']
    
#%%
def gen_prediction_input(num : int, train_or_test: str, muscle_SO2_used:float, SO2_used : list):
    for id in range(num):
        for blc in bloodConc:
            prediction_input = np.zeros((len(SO2_used),4*len(wavelength)+4)) # T1_large_SDS1/SDS2 T1_small_SDS1/SDS2 T2_large_SDS1/SDS2 T2_small_SDS1/SDS2 bloodConc id and muscle_change
            for i, s in enumerate(SO2_used):
                if (abs(s-0.7) < abs(muscle_SO2_used-base_muscle_SO2)) | (((muscle_SO2_used-base_muscle_SO2)*(s-0.7) < 0)) : # to avoid muscle SO2 change larger than IJV SO2 chang
                    continue
                else:
                    surrogate_result_T1 = pd.read_csv(os.path.join("dataset", "surrogate_result", train_or_test, 
                                                                f'bloodConc_{blc}', f'muscle_SO2_{base_muscle_SO2}', 'ijv_SO2_0.7', f'{id}_{train_or_test}.csv'))
                    surrogate_result_T2 = pd.read_csv(os.path.join("dataset", "surrogate_result", train_or_test, 
                                                                f'bloodConc_{blc}', f'muscle_SO2_{muscle_SO2_used}', f'ijv_SO2_{s}', f'{id}_{train_or_test}.csv'))
                    prediction_input[i][:20] = surrogate_result_T1['largeIJV_SDS1'] / surrogate_result_T1['largeIJV_SDS2']
                    prediction_input[i][20:40] = surrogate_result_T1['smallIJV_SDS1'] / surrogate_result_T1['smallIJV_SDS2']     
                    prediction_input[i][40:60] = surrogate_result_T2['largeIJV_SDS1'] / surrogate_result_T2['largeIJV_SDS2']
                    prediction_input[i][60:80] = surrogate_result_T2['smallIJV_SDS1'] / surrogate_result_T2['smallIJV_SDS2']
                    prediction_input[i][80] = blc
                    prediction_input[i][81] = s-0.7 # answer
                    prediction_input[i][82] = id # for analyzing used
                    prediction_input[i][83] = muscle_SO2_used-base_muscle_SO2 # muslce_mua_change

            row = np.where((np.zeros((1,4*len(wavelength)+4))==prediction_input).all(axis=1))
            prediction_input_delete = np.delete(prediction_input, row, 0)
            # print(f'origin_shape : {prediction_input.shape}, after_shape : {prediction_input_delete.shape}....')
            
            np.save(os.path.join("dataset", "prediction_result", train_or_test, f"{id}_blc_{blc}_muscle_change_{muscle_SO2_used-base_muscle_SO2:.3f}.npy"), prediction_input_delete)
            df = pd.DataFrame(prediction_input_delete)
            df.to_csv(os.path.join("dataset", "prediction_result", train_or_test, f"{id}_blc_{blc}_muscle_change_{muscle_SO2_used-base_muscle_SO2:.3f}.csv"))

if __name__ == "__main__":
    train_num = 500
    test_num = 50
    base_muscle_SO2 = 0.7
    #%%
    os.makedirs(os.path.join("dataset", "prediction_result", "train"), exist_ok=True)
    os.makedirs(os.path.join("dataset", "prediction_result", "test"), exist_ok=True)
    # gen_prediction_input(train_num, 'train', train_SO2)
    
    for muscle_SO2_used in muscle_SO2:
        print(f'now processing muscle_mua_change_from_{base_muscle_SO2}_to_{muscle_SO2_used}...')
        gen_prediction_input(train_num, 'train', muscle_SO2_used, test_SO2)
    
    for muscle_SO2_used in muscle_SO2:
        print(f'now processing muscle_mua_change_from_{base_muscle_SO2}_to_{muscle_SO2_used}...')
        gen_prediction_input(test_num, 'test', muscle_SO2_used, test_SO2)
    
    