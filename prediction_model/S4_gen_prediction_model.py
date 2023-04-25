import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd

with open('ANN_large_output.pkl', 'rb') as f:
        ANN_large_output = pickle.load(f)
with open('ANN_small_output.pkl', 'rb') as f:
        ANN_small_output = pickle.load(f)
        
condition = 200000 - 20
# SO2 = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
SO2 = [i/100 for i in range(40,95,5)]
used_wl = 20
Rmax_Rmin = {}
for i in range(condition):
    for s in SO2:
        Rmax_Rmin[f'condition_{i}_SO2_{s}'] = np.zeros((41+used_wl*10+1+20))

input_baseline = 60
for i in range(condition):
    for s in SO2:
        ANN_small_output[f'condition_{i}_SO2_{s}'][20:40] = ANN_small_output[f'condition_{i}_SO2_{s}'][:20] / ANN_small_output[f'condition_{i}_SO2_{s}'][20:40]  # SDS1 / SDS2 to make blood more important
        ANN_large_output[f'condition_{i}_SO2_{s}'][20:40] = ANN_large_output[f'condition_{i}_SO2_{s}'][:20] / ANN_large_output[f'condition_{i}_SO2_{s}'][20:40] 

        Rmax_Rmin[f'condition_{i}_SO2_{s}'][:40] = np.log(ANN_large_output[f'condition_{i}_SO2_{s}'][:40]/ANN_small_output[f'condition_{i}_SO2_{s}'][:40])
        Rmax_Rmin[f'condition_{i}_SO2_{s}'][40] = s
        Rmax_Rmin[f'condition_{i}_SO2_{s}'][41:242] = ANN_large_output[f'condition_{i}_SO2_{s}'][41:242] #small and large are same

prediction_ANN_input = {}
# SO2.pop(SO2.index(0.7))
for i in range(condition):
    for s in SO2:
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'] = np.zeros((41+used_wl*10+20+40+40+40+40+1))
for i in range(condition):
    for s in SO2:
        # delta_log = Rmax_Rmin[f'condition_{i}_SO2_{s}'][:-1] - Rmax_Rmin[f'condition_{i}_SO2_{0.7}'][:-1]
        # seperate_SDS = delta_log[20:40] / delta_log[:20] 
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][:40] = Rmax_Rmin[f'condition_{i}_SO2_{s}'][:40] - Rmax_Rmin[f'condition_{i}_SO2_{0.7}'][:40]
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][40] =  s-0.7
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][41:242] = Rmax_Rmin[f'condition_{i}_SO2_{s}'][41:242] # T2 parameter used
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][242:262] = Rmax_Rmin[f'condition_{i}_SO2_{0.7}'][201:221] # 70% IJV mua used
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][262:302] = ANN_large_output[f'condition_{i}_SO2_{s}'][:40] # T2 large
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][302:342] = ANN_small_output[f'condition_{i}_SO2_{s}'][:40] # T2 small
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][342:382] = ANN_large_output[f'condition_{i}_SO2_{0.7}'][:40] # T1 large
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][382:422] = ANN_small_output[f'condition_{i}_SO2_{0.7}'][:40] # T1 small
        
prediction_ANN_input = np.transpose(pd.DataFrame(prediction_ANN_input).to_numpy())
np.save("prediction_ANN_input.npy", prediction_ANN_input)


#%% ANN testdata
with open('test_ANN_large_output.pkl', 'rb') as f:
        ANN_large_output = pickle.load(f)
with open('test_ANN_small_output.pkl', 'rb') as f:
        ANN_small_output = pickle.load(f)
        
condition = 20000 - 20
# SO2 = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
SO2 = [i/100 for i in range(40,91,1)]
used_wl = 20
Rmax_Rmin = {}
for i in range(condition):
    for s in SO2:
        Rmax_Rmin[f'condition_{i}_SO2_{s}'] = np.zeros((41+used_wl*10+1))

for i in range(condition):
    for s in SO2:
        ANN_small_output[f'condition_{i}_SO2_{s}'][20:40] = ANN_small_output[f'condition_{i}_SO2_{s}'][:20] / ANN_small_output[f'condition_{i}_SO2_{s}'][20:40]  # SDS1 / SDS2 to make blood more important
        ANN_large_output[f'condition_{i}_SO2_{s}'][20:40] = ANN_large_output[f'condition_{i}_SO2_{s}'][:20] / ANN_large_output[f'condition_{i}_SO2_{s}'][20:40] 
        
        Rmax_Rmin[f'condition_{i}_SO2_{s}'][:40] = np.log(ANN_large_output[f'condition_{i}_SO2_{s}'][:40]/ANN_small_output[f'condition_{i}_SO2_{s}'][:40])
        Rmax_Rmin[f'condition_{i}_SO2_{s}'][40] = s
        Rmax_Rmin[f'condition_{i}_SO2_{s}'][41:242] = ANN_large_output[f'condition_{i}_SO2_{s}'][41:242] #small and large are same

prediction_ANN_input = {}
# SO2.pop(SO2.index(0.7))
for i in range(condition):
    for s in SO2:
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'] = np.zeros((41+used_wl*10+20+40+40+40+40+1))
for i in range(condition):
    for s in SO2:
        # delta_log = Rmax_Rmin[f'condition_{i}_SO2_{s}'][:-1] - Rmax_Rmin[f'condition_{i}_SO2_{0.7}'][:-1]
        # seperate_SDS = delta_log[20:40] / delta_log[:20] 
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][:40] = Rmax_Rmin[f'condition_{i}_SO2_{s}'][:40] - Rmax_Rmin[f'condition_{i}_SO2_{0.7}'][:40]
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][40] =  s-0.7
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][41:242] = Rmax_Rmin[f'condition_{i}_SO2_{s}'][41:242] # T2 parameter used
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][242:262] = Rmax_Rmin[f'condition_{i}_SO2_{0.7}'][201:221] # 70% IJV mua used
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][262:302] = ANN_large_output[f'condition_{i}_SO2_{s}'][:40] # T2 large
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][302:342] = ANN_small_output[f'condition_{i}_SO2_{s}'][:40] # T2 small
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][342:382] = ANN_large_output[f'condition_{i}_SO2_{0.7}'][:40] # T1 large
        prediction_ANN_input[f'condition_{i}_SO2_{s-0.7:.2f}'][382:422] = ANN_small_output[f'condition_{i}_SO2_{0.7}'][:40] # T1 small
        
prediction_ANN_input = np.transpose(pd.DataFrame(prediction_ANN_input).to_numpy())
np.save("test_prediction_ANN_input.npy", prediction_ANN_input)