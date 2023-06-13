from torch.utils.data import Dataset
import torch
from glob import glob
import os
import numpy as np


class myDataset(Dataset):
    def __init__(self, folder:str, num_of_id:int, bloodConc:list, num_of_blc:int, num_of_SO2:int):
        super().__init__()
        self.folder = folder
        self.files = glob(os.path.join(self.folder, "*"))
        self.files = self.files[:num_of_id*num_of_blc]
        
        self.x = np.zeros((num_of_id*num_of_blc*num_of_SO2, 40))
        self.y = np.zeros((num_of_id*num_of_blc*num_of_SO2, 1))
        self.id = np.zeros((num_of_id*num_of_blc*num_of_SO2, 1))
        for i, file in enumerate(self.files):
            data = np.load(file)
            self.x[i*num_of_SO2:i*num_of_SO2+num_of_SO2] = data[:, :40]
            self.y[i*num_of_SO2:i*num_of_SO2+num_of_SO2] = data[:, 41].reshape(-1,1)
            self.id[i*num_of_SO2:i*num_of_SO2+num_of_SO2] = data[:, 42].reshape(-1,1)
        
        # self.x[:, 80] = (self.x[:, 80] - min(bloodConc)) / (max(bloodConc) - min(bloodConc)) # normalize blc to 0~1
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        
        self.n_samples = self.y.shape[0]
    
    def __getitem__(self, index) :
        
        return self.x[index], self.y[index], self.id[index]
    
    def __len__(self):
        
        return self.n_samples