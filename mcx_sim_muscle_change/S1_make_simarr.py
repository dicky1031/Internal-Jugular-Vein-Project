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


