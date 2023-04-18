import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

#%% data preprocessing
class dataload(Dataset):
    def __init__(self, root, mus_set_path, mua_set_path, SDS1, SDS2):
        xy = np.load(root)
        self.mus_set = np.load(mus_set_path)
        self.mua_set = np.load(mua_set_path)
        self.x = torch.from_numpy(xy[:,:10])
        max_mus = np.max(self.mus_set, axis=0)[:5]
        max_mua = np.max(self.mua_set, axis=0)[:5]
        self.x_max = torch.from_numpy(np.concatenate((max_mus,max_mua)))
        min_mus = np.min(self.mus_set, axis=0)[:5]
        min_mua = np.min(self.mua_set, axis=0)[:5]
        self.x_min = torch.from_numpy(np.concatenate((min_mus,min_mua)))
        self.x = (self.x - self.x_min) / (self.x_max - self.x_min)
        self.y = torch.from_numpy(xy[:,[SDS1+9,SDS2+9]]) # SDS2:10.00mm  SDS16: 20.00mm
        self.y = -torch.log(self.y)
        self.n_samples = xy.shape[0]
                
    def __getitem__(self, index):
        
        return self.x[index], self.y[index]
        
    def __len__(self):
        
        return self.n_samples

def data_preprocess(dataset, batch_size, test_split, shuffle_dataset, random_seed):
    # create data indice for training and testing splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # count out split size
    split = int(np.floor(test_split*dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:],indices[:split]

    # creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, test_loader