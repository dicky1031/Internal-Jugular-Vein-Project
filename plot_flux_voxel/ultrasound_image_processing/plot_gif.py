#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:45:38 2023

@author: md703
"""

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
#%%
images = []

ijv_large = imageio.imread("ctchen_20220509_IJVLarge.png")
ijv_large = ijv_large[100:645]
plt.imshow(ijv_large)
images.append(ijv_large)

ijv_small = imageio.imread("ctchen_20220509_IJVSmall.png")
ijv_small = ijv_small[100:622]
ijv_small = cv2.resize(ijv_small, (502,545))
plt.imshow(ijv_small)
images.append(ijv_small)


imageio.mimsave('ctchen_IJV.gif', images, duration=0.5)

#%%
images = []

# small = np.load("ctchen_perturbed_small.npy")
# small = small[int(528//2), :, :].T
# plt.imshow(small)
# plt.axis('off')
# images.append(small)

# large = np.load("ctchen_perturbed_large.npy")
# large = large[int(528//2), :, :].T
# plt.imshow(large)
# plt.axis('off')
# images.append(large)

ijv_large = imageio.imread("ctchen_seg_large.png")
plt.imshow(ijv_large)
images.append(ijv_large)

ijv_small = imageio.imread("ctchen_seg_small.png")
plt.imshow(ijv_small)
images.append(ijv_small)

imageio.mimsave('ctchen_seg_IJV.gif', images, duration=0.5)

