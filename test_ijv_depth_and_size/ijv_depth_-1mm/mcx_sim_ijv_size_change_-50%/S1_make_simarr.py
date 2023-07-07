# %%
# run 20 wavelength including high scatter, medium scatter, low scatter for both large ijv and small ijv
# overall simulations : 20 * 3 * 2 = 120

# save file like this:

# result
#     --KB
#         -- large_to_small
#             --high
#                 --700 nm
#                 ..
#             --medium
#             --low
#         -- small_to_large


# %%
import json
import os
import numpy as np
import pandas as pd
import sys
# %% move to current file path
os.chdir(sys.path[0])
# %%
def Get_Mus_Set(mus_spectrum : json, rank : int) -> np:
    skin_mus = mus_spectrum['skin']
    subcuit_mus = mus_spectrum['fat']
    muscle_mus = mus_spectrum['muscle']
    vessel_mus = mus_spectrum['blood']
    wl = list(skin_mus.keys())
    
    # skin, fat, muscle, ijv, cca --> ijv,cca have same mus
    mus_set = np.zeros((len(wl), 5))
    for id, used_wl in enumerate(wl):
        mus_set[id][0] = skin_mus[used_wl][rank]
        mus_set[id][1] = subcuit_mus[used_wl][rank]
        mus_set[id][2] = muscle_mus[used_wl][rank]
        mus_set[id][3] = vessel_mus[used_wl][rank]
        mus_set[id][4] = vessel_mus[used_wl][rank]
    return mus_set


# %%
with open(os.path.join("OPs_used","mus_spectrum.json"), 'r') as f:
    mus_spectrum = json.load(f) 
    
mus_set = Get_Mus_Set(mus_spectrum=mus_spectrum, rank=5) # for 0 to 20 is highest scatter to lowest scatter
np.save(os.path.join('OPs_used', 'high_mus_set.npy'), mus_set)
mus_set = pd.DataFrame(
    mus_set, columns=['skin', 'fat', 'muscle', 'ijv', 'cca'])
mus_set.to_csv(os.path.join('OPs_used', 'high_mus_set.csv'), index=False)

mus_set = Get_Mus_Set(mus_spectrum=mus_spectrum, rank=10) # for 0 to 20 is highest scatter to lowest scatter
np.save(os.path.join('OPs_used', 'medium_mus_set.npy'), mus_set)
mus_set = pd.DataFrame(
    mus_set, columns=['skin', 'fat', 'muscle', 'ijv', 'cca'])
mus_set.to_csv(os.path.join('OPs_used', 'medium_mus_set.csv'), index=False)

mus_set = Get_Mus_Set(mus_spectrum=mus_spectrum, rank=15) # for 0 to 20 is highest scatter to lowest scatter
np.save(os.path.join('OPs_used', 'low_mus_set.npy'), mus_set)
mus_set = pd.DataFrame(
    mus_set, columns=['skin', 'fat', 'muscle', 'ijv', 'cca'])
mus_set.to_csv(os.path.join('OPs_used', 'low_mus_set.csv'), index=False)

# %%
with open(os.path.join("OPs_used","mua_spectrum.json"), 'r') as f:
    mua_spectrum = json.load(f) 
with open(os.path.join("OPs_used","bloodConc.json"), 'r') as f:
    bloodConc = json.load(f) 
    bloodConc = bloodConc['bloodConc']
with open(os.path.join("OPs_used","SO2.json"), 'r') as f:
    SO2 = json.load(f) 
    test_SO2 = SO2['test_SO2']
with open(os.path.join("OPs_used","wavelength.json"), 'r') as f:
    wavelength = json.load(f) 
    wavelength = wavelength['wavelength']
# load muscle_with_blodd mua spectrum

# %%
skin_mua = mua_spectrum['skin']
fat_mua = mua_spectrum['fat']
muscle_mua = mua_spectrum['muscle']
CCA_mua = mua_spectrum['cca']

# 00 0~20
# 01 20~40 1*20 20+1*20
# 02 40~60 2*20 20+2*20
# 03 60~80
# 04 80~100
# 10 100~120 1*(20*5)+0*20 20+1*(20*5)+0*20
# 11 120~140 1*(20*5)+1*20 20

for wl_idx, wl in enumerate(wavelength):
    mua_set = np.zeros((20*len(bloodConc)*len(test_SO2), 7))
    for bc_idx, bc in enumerate(bloodConc):
        for SO2_idx, using_SO2 in enumerate(test_SO2):
            mua_set[SO2_idx*20 + bc_idx*len(test_SO2)*20 : 20 + SO2_idx*20 + bc_idx*len(test_SO2)*20 ,0] = skin_mua[f'{wl}nm']
            mua_set[SO2_idx*20 + bc_idx*len(test_SO2)*20 : 20 + SO2_idx*20 + bc_idx*len(test_SO2)*20  ,1] = fat_mua[f'{wl}nm']
            mua_set[SO2_idx*20 + bc_idx*len(test_SO2)*20 : 20 + SO2_idx*20 + bc_idx*len(test_SO2)*20  ,2] = muscle_mua[f'{wl}nm']
            mua_set[SO2_idx*20 + bc_idx*len(test_SO2)*20 : 20 + SO2_idx*20 + bc_idx*len(test_SO2)*20  ,4] = CCA_mua[f'{wl}nm']
            
            mua_set[SO2_idx*20 + bc_idx*len(test_SO2)*20 : 20 + SO2_idx*20 + bc_idx*len(test_SO2)*20  ,3] = mua_spectrum[f'ijv_bloodConc_{bc}_bloodSO2_{using_SO2}'][wl_idx]
            mua_set[SO2_idx*20 + bc_idx*len(test_SO2)*20 : 20 + SO2_idx*20 + bc_idx*len(test_SO2)*20  ,5] = bc
            mua_set[SO2_idx*20 + bc_idx*len(test_SO2)*20 : 20 + SO2_idx*20 + bc_idx*len(test_SO2)*20  ,6] = using_SO2

    np.save(os.path.join('OPs_used', f'{wl}nm_mua_set.npy'), mua_set)
    mua_set = pd.DataFrame(
        mua_set, columns=['skin', 'fat', 'muscle', 'ijv', 'cca', 'bloodConc', 'ijv_SO2'])
    mua_set.to_csv(os.path.join('OPs_used', f'{wl}nm_mua_set.csv'), index=False)



