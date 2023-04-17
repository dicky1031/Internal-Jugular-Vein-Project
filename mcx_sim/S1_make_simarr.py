# %%
import json
import os
import numpy as np
import sys
import pandas as pd
# %% move to current file path
os.chdir(sys.path[0])


def Get_Mus_Set(bound, split_point, tissues):
    assert len(bound) != len(
        split_point), 'mus tissues and split size not match'

    mus_bound = np.zeros((len(tissues), 2))  # upbound and lower bound
    for i, tissue in enumerate(tissues):
        mus_bound[i] = bound[tissue]

    total_combinations = 1
    layer = []
    for i, point in enumerate(split_point):
        total_combinations *= point
        layer.append(
            list(np.linspace(mus_bound[i][0], mus_bound[i][1], point)))

    # skin, fat, muscle, ijv, cca --> ijv,cca have same mus
    mus_set = np.zeros((total_combinations, 5))
    id = 0
    for skin_mus in layer[0]:
        for subcuit_mus in layer[1]:
            for muscle_mus in layer[2]:
                for vessel_mus in layer[3]:
                    mus_set[id][0] = skin_mus
                    mus_set[id][1] = subcuit_mus
                    mus_set[id][2] = muscle_mus
                    mus_set[id][3] = vessel_mus
                    mus_set[id][4] = vessel_mus
                    id += 1

    return mus_set


def Get_Mua_Set(bound, split_point, tissues):
    assert len(bound) != len(
        split_point), 'mus tissues and split size not match'

    mua_bound = np.zeros((len(tissues), 2))  # upbound and lower bound
    for i, tissue in enumerate(tissues):
        mua_bound[i] = bound[tissue]

    total_combinations = 1
    layer = []
    for i, point in enumerate(split_point):
        total_combinations *= point
        layer.append(
            list(np.linspace(mua_bound[i][0], mua_bound[i][1], point)))

    mua_set = np.zeros((total_combinations, len(tissues)))
    id = 0
    for skin_mua in layer[0]:
        for subcuit_mua in layer[1]:
            for muscle_mua in layer[2]:
                for ijv_mua in layer[3]:
                    for cca_mua in layer[4]:
                        mua_set[id] = [skin_mua, subcuit_mua,
                                       muscle_mua, ijv_mua, cca_mua]
                        id += 1

    return mua_set


if __name__ == "__main__":
    # Get Mus set
    with open(os.path.join("OPs_used", "mus_bound.json"), "r") as f:
        mus_bound = json.load(f)
    mus_tissues = ['skin', 'fat', 'muscle', 'blood']
    split_point = [7, 7, 5, 5]
    mus_set = Get_Mus_Set(mus_bound, split_point, mus_tissues)
    np.save(os.path.join('OPs_used', 'mus_set.npy'), mus_set)
    mus_set = pd.DataFrame(
        mus_set, columns=['skin', 'fat', 'muscle', 'ijv', 'cca'])
    mus_set.to_csv(os.path.join('OPs_used', 'mus_set.csv'))

    # Get Mua ser
    with open(os.path.join("OPs_used", "mua_bound.json"), "r") as f:
        mua_bound = json.load(f)
    mua_tissues = ['skin', 'fat', 'muscle', 'ijv', 'cca']
    split_point = [3, 3, 5, 7, 7]
    mua_set = Get_Mua_Set(mua_bound, split_point, mua_tissues)
    np.save(os.path.join('OPs_used', 'mua_set.npy'), mua_set)
    mua_set = pd.DataFrame(mua_set, columns=mua_tissues)
    mua_set.to_csv(os.path.join('OPs_used', 'mua_set.csv'))

    # load mua for calculating reflectance
    muaPath = os.path.join("input_template", "mua_test.json")
    with open(muaPath) as f:
        mua = json.load(f)
    mua["4: Skin"] = np.mean(mua_bound['skin'])
    mua["5: Fat"] = np.mean(mua_bound['fat'])
    mua["6: Muscle"] = np.mean(mua_bound['muscle'])
    mua["7: Muscle or IJV (Perturbed Region)"] = np.mean(
        mua_bound['ijv'])  # IJV
    mua["8: IJV"] = np.mean(mua_bound['ijv'])
    mua["9: CCA"] = np.mean(mua_bound['cca'])

    with open(muaPath, "w") as f:
        json.dump(mua, f, indent=4)
