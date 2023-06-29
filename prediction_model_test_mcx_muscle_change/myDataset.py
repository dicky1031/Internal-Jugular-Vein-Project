from torch.utils.data import Dataset
import torch
from glob import glob
import os
import numpy as np


class myDataset(Dataset):
    def __init__(self, folder:str):
        super().__init__()
        self.folder = folder
        self.files = glob(os.path.join(self.folder, "*.npy"))
        
        self.x = []
        self.y = []
        self.id = []
        self.mua_rank = []
        self.muscle_SO2 = []
        for file in self.files:
            datas = np.load(file, allow_pickle=True)
            for data in datas:
                self.x.append(data[:40])
                self.y.append(data[[41]])
                self.id.append(data[42])
                self.mua_rank.append(data[43])
                self.muscle_SO2.append(data[44])
        self.x = np.array(self.x, dtype=np.float64)
        self.y = np.array(self.y, dtype=np.float64)
        self.id = np.array(self.id)
        self.mua_rank = np.array(self.mua_rank)
        self.muscle_SO2 = np.array(self.muscle_SO2, dtype=np.float64)
        
        # self.x[:, 80] = (self.x[:, 80] - min(bloodConc)) / (max(bloodConc) - min(bloodConc)) # normalize blc to 0~1
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        
        self.n_samples = self.y.shape[0]
    
    def __getitem__(self, index) :
        
        return self.x[index], self.y[index], self.id[index], self.mua_rank[index], self.muscle_SO2[index]
    
    def __len__(self):
        
        return self.n_samples