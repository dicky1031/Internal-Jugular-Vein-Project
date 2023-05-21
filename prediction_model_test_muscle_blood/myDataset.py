from torch.utils.data import Dataset
import torch
from glob import glob
import os
import numpy as np


class myDataset(Dataset):
    def __init__(self, folder:str, num_of_id:int, bloodConc:list, num_of_blc:int, num_of_SO2:int, num_of_muscle_SO2:int):
        super().__init__()
        self.folder = folder
        self.files = glob(os.path.join(self.folder, "*.npy"))
        
        self.x = []
        self.y = []
        self.id = []
        self.muscle_SO2 = []
        for file in self.files:
            datas = np.load(file)
            for data in datas:
                self.x.append(data[:81])
                self.y.append(data[[81,83]])
                self.id.append(data[82])
                self.muscle_SO2.append(data[83])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.id = np.array(self.id)
        self.muscle_SO2 = np.array(self.muscle_SO2)
        
        # self.x[:, 80] = (self.x[:, 80] - min(bloodConc)) / (max(bloodConc) - min(bloodConc)) # normalize blc to 0~1
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        
        self.n_samples = self.y.shape[0]
    
    def __getitem__(self, index) :
        
        return self.x[index], self.y[index], self.id[index], self.muscle_SO2[index]
    
    def __len__(self):
        
        return self.n_samples