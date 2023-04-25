
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from Preprocessing import dataload
from surrogate_model import ANN
import os
import sys
os.chdir(sys.path[0])

os.path.join()


if __name__ == "__main__":
    condition = 200000 - 20
    used_wl = 20
    #%% small model train
    ijv_size = "small"
    SO2 = [i/100 for i in range(40,95,5)]
    with open('large_sim_dataset.pkl', 'rb') as f:
        ANN_train_input = pickle.load(f)
    prediction_ANN_output = {}
    for i in range(condition):
        for s in SO2:
            prediction_ANN_output[f'condition_{i}_SO2_{s}'] = np.zeros(41+used_wl*10+1)  #[SDS1_reflectance_wl1....wl_last, SDS2_wl1...wl_last, ans]
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"{ijv_size}_ANN_model.pth"))
    count = 0
    for i in range(condition):
        print(f'processing {i}')
        for s in SO2: 
            root = ANN_train_input[f'condition_{i}_SO2_{s}']
            mus_set_path = "mus_set.npy"
            mua_set_path = "mua_set.npy"
            data, target, bloodConc = dataload(root, mus_set_path, mua_set_path) # data : wl*optical_parameter = 20*10
            reflectance = model(data.to(torch.float32).cuda())
            reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][:20] = reflectance[:,0] # SDS1
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][20:40] = reflectance[:,1] # SDS2
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target.unique()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][41:61] = root[:,0] # skin mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][61:81] = root[:,1] # fat mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][81:101] = root[:,2] # muscle mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][101:121] = root[:,3] # IJV mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][121:141] = root[:,4] # CCA mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][141:161] = root[:,5] # skin mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][161:181] = root[:,6] # fat mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][181:201] = root[:,7] # muscle mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][201:221] = root[:,8] # IJV mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][221:241] = root[:,9] # CCA mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][241] = bloodConc.unique()
            
            # for wl,(data, target) in enumerate(dataset):
            #     break
            #     reflectance = model(data.to(torch.float32).cuda())
            #     reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = reflectance[0] # SDS1
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl+20] = reflectance[1] # SDS2
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = root[wl][:-1]
            #     # prediction_ANN_output['optical parameter'] = 
            #     # prediction_ANN_output[count,wl] = reflectance
            #     # prediction_ANN_output[count,-1] = target
            count += 1
    with open(f'ANN_{ijv_size}_output.pkl', 'wb') as f:
        pickle.dump(prediction_ANN_output, f)
        
    #%% small model test
    condition = 20000 - 20
    used_wl = 20
    ijv_size = "small"
    SO2 = [i/100 for i in range(40,91,1)]
    with open('test_large_sim_dataset.pkl', 'rb') as f:
        ANN_test_input = pickle.load(f)
    prediction_ANN_output = {}
    for i in range(condition):
        for s in SO2:
            prediction_ANN_output[f'condition_{i}_SO2_{s}'] = np.zeros(41+used_wl*10+1)  #[SDS1_reflectance_wl1....wl_last, SDS2_wl1...wl_last, ans]
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"{ijv_size}_ANN_model.pth"))
    count = 0
    for i in range(condition):
        print(f'processing {i}')
        for s in SO2: 
            root = ANN_test_input[f'condition_{i}_SO2_{s}']
            mus_set_path = "mus_set.npy"
            mua_set_path = "mua_set.npy"
            # dataset = dataload(root, mus_set_path, mua_set_path)
            data, target, bloodConc = dataload(root, mus_set_path, mua_set_path) # data : wl*optical_parameter = 20*10
            reflectance = model(data.to(torch.float32).cuda())
            reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][:20] = reflectance[:,0] # SDS1
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][20:40] = reflectance[:,1] # SDS2
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target.unique()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][41:61] = root[:,0] # skin mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][61:81] = root[:,1] # fat mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][81:101] = root[:,2] # muscle mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][101:121] = root[:,3] # IJV mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][121:141] = root[:,4] # CCA mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][141:161] = root[:,5] # skin mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][161:181] = root[:,6] # fat mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][181:201] = root[:,7] # muscle mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][201:221] = root[:,8] # IJV mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][221:241] = root[:,9] # CCA mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][241] = bloodConc.unique()
            
            # for wl,(data, target) in enumerate(dataset):
            #     break
            #     reflectance = model(data.to(torch.float32).cuda())
            #     reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = reflectance[0] # SDS1
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl+20] = reflectance[1] # SDS2
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = root[wl][:-1]
            #     # prediction_ANN_output['optical parameter'] = 
            #     # prediction_ANN_output[count,wl] = reflectance
            #     # prediction_ANN_output[count,-1] = target
            count += 1
    with open(f'test_ANN_{ijv_size}_output.pkl', 'wb') as f:
        pickle.dump(prediction_ANN_output, f)
        
    #%% train large model
    condition = 200000 - 20
    used_wl = 20
    ijv_size = "large"
    SO2 = [i/100 for i in range(40,95,5)]
    # with open('large_sim_dataset.pkl', 'rb') as f:
    #     ANN_input = pickle.load(f)
    prediction_ANN_output = {}
    for i in range(condition):
        for s in SO2:
            prediction_ANN_output[f'condition_{i}_SO2_{s}'] = np.zeros(41+used_wl*10+1)  #[SDS1_reflectance_wl1....wl_last, SDS2_wl1...wl_last, ans]
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"{ijv_size}_ANN_model.pth"))
    count = 0
    for i in range(condition):
        print(f'processing {i}')
        for s in SO2: 
            root = ANN_train_input[f'condition_{i}_SO2_{s}']
            mus_set_path = "mus_set.npy"
            mua_set_path = "mua_set.npy"
            # dataset = dataload(root, mus_set_path, mua_set_path)
            data, target, bloodConc = dataload(root, mus_set_path, mua_set_path) # data : wl*optical_parameter = 20*10
            reflectance = model(data.to(torch.float32).cuda())
            reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][:20] = reflectance[:,0] # SDS1
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][20:40] = reflectance[:,1] # SDS2
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target.unique()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][41:61] = root[:,0] # skin mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][61:81] = root[:,1] # fat mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][81:101] = root[:,2] # muscle mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][101:121] = root[:,3] # IJV mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][121:141] = root[:,4] # CCA mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][141:161] = root[:,5] # skin mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][161:181] = root[:,6] # fat mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][181:201] = root[:,7] # muscle mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][201:221] = root[:,8] # IJV mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][221:241] = root[:,9] # CCA mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][241] = bloodConc.unique()
            
            # for wl,(data, target) in enumerate(dataset):
            #     break
            #     reflectance = model(data.to(torch.float32).cuda())
            #     reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = reflectance[0] # SDS1
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl+20] = reflectance[1] # SDS2
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = root[wl][:-1]
            #     # prediction_ANN_output['optical parameter'] = 
            #     # prediction_ANN_output[count,wl] = reflectance
            #     # prediction_ANN_output[count,-1] = target
            count += 1
    with open(f'ANN_{ijv_size}_output.pkl', 'wb') as f:
        pickle.dump(prediction_ANN_output, f)
    
    #%% test large model
    condition = 20000 - 20
    used_wl = 20
    ijv_size = "large"
    SO2 = [i/100 for i in range(40,91,1)]
    # with open('test_large_sim_dataset.pkl', 'rb') as f:
    #     ANN_input = pickle.load(f)
    prediction_ANN_output = {}
    for i in range(condition):
        for s in SO2:
            prediction_ANN_output[f'condition_{i}_SO2_{s}'] = np.zeros(41+used_wl*10+1)  #[SDS1_reflectance_wl1....wl_last, SDS2_wl1...wl_last, ans]
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"{ijv_size}_ANN_model.pth"))
    count = 0
    for i in range(condition):
        print(f'processing {i}')
        for s in SO2: 
            root = ANN_test_input[f'condition_{i}_SO2_{s}']
            mus_set_path = "mus_set.npy"
            mua_set_path = "mua_set.npy"
            # dataset = dataload(root, mus_set_path, mua_set_path)
            data, target, bloodConc = dataload(root, mus_set_path, mua_set_path) # data : wl*optical_parameter = 20*10
            reflectance = model(data.to(torch.float32).cuda())
            reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][:20] = reflectance[:,0] # SDS1
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][20:40] = reflectance[:,1] # SDS2
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target.unique()
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][41:61] = root[:,0] # skin mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][61:81] = root[:,1] # fat mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][81:101] = root[:,2] # muscle mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][101:121] = root[:,3] # IJV mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][121:141] = root[:,4] # CCA mus
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][141:161] = root[:,5] # skin mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][161:181] = root[:,6] # fat mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][181:201] = root[:,7] # muscle mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][201:221] = root[:,8] # IJV mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][221:241] = root[:,9] # CCA mua
            prediction_ANN_output[f'condition_{i}_SO2_{s}'][241] = bloodConc.unique()
            
            # for wl,(data, target) in enumerate(dataset):
            #     break
            #     reflectance = model(data.to(torch.float32).cuda())
            #     reflectance = torch.exp(-reflectance).detach().cpu().numpy()
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = reflectance[0] # SDS1
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl+20] = reflectance[1] # SDS2
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][40] = target
            #     prediction_ANN_output[f'condition_{i}_SO2_{s}'][wl] = root[wl][:-1]
            #     # prediction_ANN_output['optical parameter'] = 
            #     # prediction_ANN_output[count,wl] = reflectance
            #     # prediction_ANN_output[count,-1] = target
            count += 1
    with open(f'test_ANN_{ijv_size}_output.pkl', 'wb') as f:
        pickle.dump(prediction_ANN_output, f)
    
    