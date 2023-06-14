import numpy as np
import os
import sys
import torch
os.chdir(sys.path[0])

a = np.load(os.path.join("dataset", "kb", "low_scatter_prediction_input_muscle_1", "all_absorption","prediction_input.npy"), allow_pickle=True)
x = []
x.append(a[0,:])
x.append(a[1,:])
x.append(a[2,:])
x = np.array(x, dtype=np.float64)
print(x.shape)

y = torch.from_numpy(x)