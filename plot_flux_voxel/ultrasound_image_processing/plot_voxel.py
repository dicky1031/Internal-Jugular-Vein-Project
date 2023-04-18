#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 00:13:13 2022

@author: md703
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

vol = np.load('ctchen_perturbed_small_to_large.npy')
vol = vol.T
# plt.imshow(vol[int(vol.shape[0]//2),:,:].T)
vol = vol[:,:,::8]
cmap = ['red', 'salmon', 'sienna', 'silver', 'tan', 'white', 'violet', 'wheat', 'yellow']
colors = np.empty(vol.shape, dtype=object)
edgecolors = [1,1,1,0]

vol = explode(vol)

colors = np.empty(list(vol.shape) + [4], dtype=np.float32)
alpha = 0.3
colors[vol==1] = [1, 0, 0, alpha]
colors[vol==2] = [0, 1, 0, alpha]
colors[vol==3] = [0, 0, 1, alpha]
colors[vol==4] = [1, 1, 0, alpha]
colors[vol==5] = [1, 0, 1, alpha]
colors[vol==6] = [0, 1, 1, 0.005]
colors[vol==7] = [1, 1, 1, 1]
colors[vol==8] = [0, 0, 0, 1]
colors[vol==9] = [0.5, 0.5, 0.5, 1]
# colors[1] = [0, 1, 0, alpha]
# colors[2] = [0, 0, 1, alpha]
# colors[3] = [1, 1, 0, alpha]
# colors[4] = [0, 1, 1, alpha]

x, y, z = np.indices(np.array(vol.shape) + 1).astype(float) // 2
x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, vol, facecolors=colors, edgecolors=edgecolors)
ax.view_init(elev=25, azim=135)
ax.axis('off')
plt.show()



