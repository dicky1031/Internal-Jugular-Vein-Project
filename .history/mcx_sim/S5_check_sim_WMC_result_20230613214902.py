import os
import numpy as np
import sys

# %% move to current file path
os.chdir(sys.path[0])

MUS_SET = np.load(os.path.join("OPs_used", "mus_set.npy"))


def MC_check(folder):
    record = []
    for i in range(1, MUS_SET.shape[0]+1):
        filepath = os.path.join(
            folder, f"run_{i}", "post_analysis", f"run_{i}_simulation_result.json")
        if not os.path.isfile(filepath):
            record.append(i)
    if record != []:
        print(f"{folder} MC sim...")
        for i in record:
            print(f"run_{i} ", end=" ")
        print("doesn`t exist!")
    else:
        print(f"{folder} MC sim all complete")


def WMC_check(folder):
    record = []
    nan = []
    for i in range(1, MUS_SET.shape[0]+1):
        filepath = os.path.join(folder, f"mus_{i}.npy")
        if not os.path.isfile(filepath):
            record.append(i)
        else:
            data = np.load(filepath)
            if np.isnan(data).any():
                nan.append(i)
    if record != []:
        print(f"{folder} WMC sim...")
        for i in record:
            print(f"run_{i} ", end=" ")
        print("doesn`t exist!")
    else:
        print(f"{folder} WMC sim all complete")

    if nan != []:
        for i in nan:
            print(f"run_{i}", end=" ")
        print("has nan!")


if __name__ == "__main__":
    result_folder = "ctchen"
    subject = "ctchen"

    folder = os.path.join("dataset", result_folder,
                          f"{subject}_ijv_small_to_large")
    MC_check(folder)
    folder = os.path.join("dataset", result_folder,
                          f"{subject}_ijv_large_to_small")
    MC_check(folder)
    folder = os.path.join("dataset", result_folder, f"{subject}_dataset_large")
    WMC_check(folder)
    older = os.path.join("dataset", result_folder, f"{subject}_dataset_small")
    WMC_check(folder)
