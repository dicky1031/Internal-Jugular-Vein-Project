import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import json

folder = "ctchen_1e9_ijv_large_to_small"
mus_start = 1
mus_end = 1225
used_photon = 1e9
SDS = ['sds_1', 'sds_15']
CV = np.zeros((len(SDS), mus_end-mus_start+1))
num_photons = np.zeros((mus_end-mus_start+1))
used_mus = np.zeros((4, mus_end-mus_start+1))
if not os.path.isdir(os.path.join("pic", folder)):
    os.mkdir(os.path.join("pic", folder))
topK = 50
tissue = ['Skin', 'Muscle', 'Fat', 'IJV']

if __name__ == '__main__':
    for i in range(mus_start, mus_end+1):
        model_path = os.path.join(
            folder, "LUT", f"run_{i}", "model_parameters.json")
        with open(model_path, 'r') as f:
            model = json.load(f)
        used_mus[0, i-1] = model['OptParam']['Skin']['mus']
        used_mus[1, i-1] = model['OptParam']['Muscle']['mus']
        used_mus[2, i-1] = model['OptParam']['Fat']['mus']
        used_mus[3, i-1] = model['OptParam']['IJV']['mus']

        path = os.path.join(
            folder, "LUT", f"run_{i}", "post_analysis", f"run_{i}_simulation_result.json")
        num = len(
            glob(os.path.join(folder, "LUT", f"run_{i}", "mcx_output", "*")))
        with open(path, 'r') as f:
            data = json.load(f)
        num_photons[i-1] = used_photon*num
        for s_idx, s in enumerate(SDS):
            CV[s_idx][i-1] = data['GroupingSampleCV'][s]/(num**0.5)

    plt.figure(figsize=(16, 8))
    plt.plot(CV[0], 'g', label='SDS : 10mm')
    plt.plot(CV[1], 'r', label='SDS : 20mm')
    plt.legend()
    plt.xlabel("simulation set")
    plt.ylabel("CV(%)")
    plt.title("CV analysis for each simulation")
    plt.savefig(os.path.join("pic", folder, "sim_CV_analysis.png"))
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.plot(num_photons)
    plt.xlabel("simulation set")
    plt.ylabel("used photons")
    plt.title("used photons for each simulation")
    plt.savefig(os.path.join("pic", folder, "sim_photons_num.png"))
    plt.show()

    min_mus = used_mus.min(axis=1).reshape(1, 4)
    min_mus = 1/min_mus
    total_influence = (np.dot(min_mus, used_mus)/4).reshape(-1)

    sort_idx = np.argsort(num_photons)[-topK:]
    sort_idx = np.flip(sort_idx)
    num_photons = num_photons[sort_idx]
    used_mus = used_mus[:, sort_idx]
    total_influence = total_influence[sort_idx]

    fig = plt.figure()
    fig.suptitle('Analysis Each Parameter influence')
    for i in range(4):
        subax = fig.add_subplot(2, 2, i+1)
        plt.xlabel('simulation set')
        subax.plot(num_photons, label='photons')
        subax.set_ylabel('used photons', color='b')
        # subax.plot(used_mus[i,:], label = f'{tissue[i]}')
        # subax.legend()
        subax2 = subax.twinx()
        subax2.plot(used_mus[i, :], 'g', label=f'{tissue[i]}')
        subax2.set_ylabel(f'{tissue[i]} $\mu_s$($mm^{-1}$)', color='g')
    plt.tight_layout()
    plt.savefig(os.path.join("pic", folder, "tissue_influence_analysis.png"))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Total parameter influence analysis")
    plt.xlabel(f'simulation top {topK} set')
    ax.plot(num_photons, label='photons')
    ax.set_ylabel('used photons', color='b')
    ax2 = ax.twinx()
    ax2.plot(total_influence, 'g')
    ax2.set_ylabel('total parameters ratio', color='g')
    plt.tight_layout()
    plt.savefig(os.path.join("pic", folder, "total_influence_analysis.png"))
    plt.show()
