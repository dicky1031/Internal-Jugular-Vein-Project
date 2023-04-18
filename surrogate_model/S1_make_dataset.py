import numpy as np
import os
from glob import glob
import sys
os.chdir(sys.path[0])


def gen_dataset(subject, ijv_size):
    assert os.path.isdir(os.path.join("dataset",
        f"{subject}_dataset_{ijv_size}")), "you have to copy the WMC result to this directory"
    assert os.path.isdir(os.path.join("OPs_used")
                         ), "you have to copy OPs_used to this directory"
    assert os.path.isfile(os.path.join(
        "OPs_used", "mus_set.npy")), "mus_set.npy is missing"
    assert os.path.isfile(os.path.join(
        "OPs_used", "mua_set.npy")), "mua_set.npy is missing"

    dataset_folder = os.path.join("dataset", f"{subject}_dataset_{ijv_size}")
    datapath = sorted(glob(os.path.join(dataset_folder, "*")),
                      key=lambda x: int(x.split("_")[-1][:-4]))
    mus_set = np.load(os.path.join("OPs_used", "mus_set.npy"))
    mua_set = np.load(os.path.join("OPs_used", "mua_set.npy"))
    # mus(5), mua(5), SDS(21)
    data = np.empty((mus_set.shape[0]*mua_set.shape[0], 31))
    for idx, path in enumerate(datapath):
        p = path.split("/")[-1]
        print(f"Now processing {p} .....")
        data[(idx)*mua_set.shape[0]:(idx+1)*mua_set.shape[0]] = np.load(path)

    return data


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    subject = 'ctchen'
    
    ijv_sizes = ['large', 'small']
    for ijv_size in ijv_sizes:
        data = gen_dataset(subject, ijv_size)
        np.save(os.path.join('dataset',f'{subject}_{ijv_size}_dataset.npy'), data)
