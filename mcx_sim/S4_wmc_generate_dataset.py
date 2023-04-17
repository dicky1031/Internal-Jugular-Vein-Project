
import numpy as np
import cupy as cp
import jdata as jd
import json
import os
from glob import glob
from tqdm import tqdm
import sys
# %% move to current file path
os.chdir(sys.path[0])

# %% Global
mua_set = np.load(os.path.join("OPs_used", "mua_set.npy"))
mus_set = np.load(os.path.join("OPs_used", "mus_set.npy"))
# hardware mua setting
air_mua = 0
PLA_mua = 10000
prism_mua = 0
mua_used = [mua_set.shape[0]*[air_mua],
            mua_set.shape[0]*[PLA_mua],
            mua_set.shape[0]*[prism_mua],
            list(mua_set[:, 0]),  # skin mua
            list(mua_set[:, 1]),  # fat mua
            list(mua_set[:, 2]),  # musle mua
            list(mua_set[:, 2]),  # perturbed region = musle
            list(mua_set[:, 3]),  # IJV mua
            list(mua_set[:, 4])  # CCA mua
            ]
# each detector has 6 copy to magnify the signal
used_SDS = cp.array([0, 1, 2, 3, 4, 5])
# %%


class post_processing:

    def __init__(self, ID):
        self.air_mua = 0
        self.PLA_mua = 10000
        self.prism_mua = 0
        self.ID = ID
        # self.used_SDS = np.array([0,1,2,3,4,5])

    def get_used_mus(self, mus_set, mus_run_idx):
        self.mus_used = [mus_set[mus_run_idx-1, 0],  # skin_mus
                         mus_set[mus_run_idx-1, 1],  # fat_mus
                         mus_set[mus_run_idx-1, 2],  # musle_mus
                         mus_set[mus_run_idx-1, 3],  # ijv_mus
                         mus_set[mus_run_idx-1, 4]  # cca_mus
                         ]
        return self.mus_used

    def get_used_mua(self, mua_set):
        if self.ID.find("small_to_large") != -1:
            self.mua_used = np.array([mua_set.shape[0]*[self.air_mua],
                                      mua_set.shape[0]*[self.PLA_mua],
                                      mua_set.shape[0]*[self.prism_mua],
                                      list(mua_set[:, 0]),  # skin mua
                                      list(mua_set[:, 1]),  # fat mua
                                      list(mua_set[:, 2]),  # musle mua
                                      # perturbed region = musle
                                      list(mua_set[:, 2]),
                                      list(mua_set[:, 3]),  # IJV mua
                                      list(mua_set[:, 4])  # CCA mua
                                      ])
        elif self.ID.find("large_to_small") != -1:
            self.mua_used = np.array([mua_set.shape[0]*[self.air_mua],
                                      mua_set.shape[0]*[self.PLA_mua],
                                      mua_set.shape[0]*[self.prism_mua],
                                      list(mua_set[:, 0]),  # skin mua
                                      list(mua_set[:, 1]),  # fat mua
                                      list(mua_set[:, 2]),  # musle mua
                                      # perturbed region = IJV mua
                                      list(mua_set[:, 3]),
                                      list(mua_set[:, 3]),  # IJV mua
                                      list(mua_set[:, 4])  # CCA mua
                                      ])
        else:
            raise Exception("Something wrong in your ID name !")
        return cp.array(self.mua_used)

    def get_data(self, mus_run_idx):
        self.session = f"run_{mus_run_idx}"
        with open(os.path.join(os.path.join(self.ID, self.session), "config.json")) as f:
            config = json.load(f)  # about detector na, & photon number
        with open(os.path.join(os.path.join(self.ID, self.session), "model_parameters.json")) as f:
            # about index of materials & fiber number
            modelParameters = json.load(f)
        self.photonNum = int(config["PhotonNum"])
        self.fiberSet = modelParameters["HardwareParam"]["Detector"]["Fiber"]
        # about paths of detected photon data
        self.detOutputPathSet = glob(os.path.join(
            config["OutputPath"], self.session, "mcx_output", "*.jdat"))
        self.detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
        self.detectorNum = len(self.fiberSet)*3*2
        # self.dataset_output = np.empty([mua_set.shape[0],10+len(fiberSet)])

        return self.photonNum, self.fiberSet, self.detOutputPathSet, self.detectorNum


def WMC(detOutputPathSet, detectorNum, used_SDS, used_mua):
    reflectance = cp.zeros((detectorNum, mua_set.shape[0]))
    group_reflectance = cp.zeros((len(fiberSet), mua_set.shape[0]))
    for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
        # main
        # sort (to make calculation of cv be consistent in each time)
        detOutput = jd.load(detOutputPath)
        info = detOutput["MCXData"]["Info"]
        photonData = detOutput["MCXData"]["PhotonData"]
        # unit conversion for photon pathlength
        photonData["ppath"] = photonData["ppath"] * info["LengthUnit"]
        photonData["detid"] = photonData["detid"] - \
            1  # shift detid from 0 to start
        for detectorIdx in range(info["DetNum"]):
            ppath = cp.asarray(
                photonData["ppath"][photonData["detid"][:, 0] == detectorIdx].astype(np.float64))
            # for split_idx in range(int(ppath.shape[0]*0.2),ppath.shape[0],int(ppath.shape[0]*0.2)): # split 20% for using less memory
            #     head_idx = split_idx - int(ppath.shape[0]*0.2)
            #     # I = I0 * exp(-mua*L)
            #     # W_sim
            #     reflectance[detOutputIdx][detectorIdx] = cp.exp(-ppath[head_idx:split_idx,:]@used_mua).sum() / photonNum

            # batch ppath for GPU use
            max_memory = 100000
            if ppath.shape[0] > max_memory:
                for idx, ppath_idx in enumerate(range(0, ppath.shape[0]//max_memory)):
                    if idx == 0:
                        batch_ppath_reflectance = cp.exp(
                            -ppath[max_memory*(ppath_idx):max_memory*(ppath_idx+1)]@used_mua).sum(axis=0)
                        # print(f'idx ={max_memory*(ppath_idx)} ~ {max_memory*(ppath_idx+1)} \n   r : {batch_ppath_reflectance}')
                    else:
                        batch_ppath_reflectance += cp.exp(-ppath[max_memory*(
                            ppath_idx):max_memory*(ppath_idx+1)]@used_mua).sum(axis=0)
                        # print(f'idx ={max_memory*(ppath_idx)} ~ {max_memory*(ppath_idx+1)} \n   r : {batch_ppath_reflectance}')
                batch_ppath_reflectance += cp.exp(-ppath[max_memory*(
                    ppath_idx+1):]@used_mua).sum(axis=0)
                # print(f'idx =\{max_memory*(ppath_idx+1)} to last \n   r : {batch_ppath_reflectance}')
            else:
                batch_ppath_reflectance = cp.exp(-ppath@used_mua).sum(axis=0)

            reflectance[detectorIdx][:] = batch_ppath_reflectance / photonNum
        for fiberIdx in range(len(fiberSet)):
            group_reflectance[fiberIdx][:] = group_reflectance[fiberIdx][:] + \
                cp.mean(reflectance[used_SDS][:], axis=0)
            used_SDS = used_SDS + 2*3

    output_R = (group_reflectance/(detOutputIdx+1)).T  # mean

    return output_R


def PMC(detOutputPathSet, detectorNum, used_SDS, used_mua):

    reflectance = cp.zero(
        (len(detOutputPathSet), detectorNum, mua_set.shape[0]))
    group_reflectance = cp.zero(
        (len(detOutputPathSet), len(fiberSet), mua_set.shape[0]))
    for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
        # main
        # sort (to make calculation of cv be consistent in each time)
        detOutput = jd.load(detOutputPath)
        info = detOutput["MCXData"]["Info"]
        photonData = detOutput["MCXData"]["PhotonData"]
        # unit conversion for photon pathlength
        photonData["ppath"] = photonData["ppath"] * info["LengthUnit"]
        photonData["detid"] = photonData["detid"] - \
            1  # shift detid from 0 to start
        for detectorIdx in range(info["DetNum"]):
            ppath = cp.asarray(
                photonData["ppath"][photonData["detid"][:, 0] == detectorIdx].astype(np.float64))
            nscat = cp.asarray(
                photonData["nscat"][photonData["detid"][:, 0] == detectorIdx].astype(np.float64))
            W_sim = cp.exp(-ppath@used_mua) / photonNum
            if ID.find("small_to_large") != -1:
                us_new = cp.asarray(
                    mus_set[mus_run_idx-1, 3].astype(np.float64))  # IJV mus
                ua_new = cp.asarray(
                    mua_set[:, 3].astype(np.float64))  # IJV mua
                us_old = cp.asarray(
                    mus_set[mus_run_idx-1, 2].astype(np.float64))  # muscle mus
                ua_old = cp.asarray(
                    mua_set[:, 2].astype(np.float64))  # muscle mua
                ut_new = cp.asarray((us_new + ua_new).astype(np.float64))
                ut_old = cp.asarray((us_old + ua_old).astype(np.float64))
            elif ID.find("large_to_small") != -1:
                us_new = cp.asarray(
                    mus_set[mus_run_idx-1, 2].astype(np.float64))  # muscle mus
                ua_new = cp.asarray(
                    mua_set[:, 2].astype(np.float64))  # muscle mua
                us_old = cp.asarray(
                    mus_set[mus_run_idx-1, 3].astype(np.float64))  # IJV mus
                ua_old = cp.asarray(
                    mua_set[:, 3].astype(np.float64))  # IJV mua
                ut_new = cp.asarray((us_new + ua_new).astype(np.float64))
                ut_old = cp.asarray((us_old + ua_old).astype(np.float64))
            else:
                raise Exception("Something wrong in your ID name !")
            ppath = ppath[:, 7]  # perturb region pathlength
            nscat = nscat[:, 7]  # perturb region # of collision

            # W_new = W_sim*((us_new/ut_new)/(us_old/ut_old))^j*(ut_new/ut_old)^j*exp(-ut_new*path)/exp(-ut_old*path)
            # --> W_new =  W_sim*[((us_new/us_old)**nscat)]*[(cp.exp(-ut_new*ppath)/cp.exp(-ut_old*ppath))]
            # --> W_new =  W_sim*[((us_new/us_old)**nscat)]*[(cp.exp(-(ut_new-ut_old)*ppath))]
            mid_term = cp.repeat(((us_new/us_old)**nscat), mua_set.shape[0]).reshape(
                nscat.shape[0], mua_set.shape[0])  # ((us_new/us_old)**nscat)
            diff_ut = (ut_new-ut_old)
            tail_term = cp.exp(-cp.outer(ppath, diff_ut))
            W_new = W_sim*mid_term*tail_term
            reflectance[detOutputIdx][detectorIdx][:] = W_new.sum(axis=0)
        for fiberIdx in range(len(fiberSet)):
            group_reflectance[detOutputIdx][fiberIdx][:] = cp.mean(
                reflectance[detOutputIdx][used_SDS][:], axis=0)
            used_SDS = used_SDS + 2*3

    output_R = group_reflectance.mean(axis=0).T

    return output_R


if __name__ == "__main__":
    # script setting
    # datasetpath = sys.argv[1] #datasetpath = "KB_dataset_small"
    # ID = sys.argv[2] # ID = "KB_ijv_small_to_large"
    # mus_start = int(sys.argv[3])
    # mus_end = int(sys.argv[4])

    result_folder = "ctchen"
    subject = "ctchen"
    ijv_type = "large_to_small"
    mus_start = 1
    mus_end = 1

    ID = os.path.join("result", result_folder, f"{subject}_ijv_{ijv_type}")
    ijv_size = ijv_type.split("_")[0]
    datasetpath = f"{subject}_dataset_{ijv_size}"
    os.makedirs(os.path.join("result", result_folder,
                datasetpath), exist_ok=True)

    processsor = post_processing(ID)
    for mus_run_idx in tqdm(range(mus_start, mus_end+1)):
        print(f"\n Now run mus_{mus_run_idx}")
        photonNum, fiberSet, detOutputPathSet, detectorNum = processsor.get_data(
            mus_run_idx)
        used_mus = processsor.get_used_mus(mus_set, mus_run_idx)
        used_mus = np.tile(np.array(used_mus), mua_set.shape[0]).reshape(
            mua_set.shape[0], 5)
        dataset_output = np.empty([mua_set.shape[0], 10+len(fiberSet)])
        used_mua = processsor.get_used_mua(mua_set)
        if datasetpath.find("small_to_large") != -1 or datasetpath.find("large_to_small") != -1:
            output_R = PMC(detOutputPathSet, detectorNum, used_SDS, used_mua)
        else:
            output_R = WMC(detOutputPathSet, detectorNum, used_SDS, used_mua)
        dataset_output[:, 10:] = cp.asnumpy(output_R)
        used_mua = used_mua[3:]  # skin, fat, muscle, perturbed, IJV, CCA
        used_mua = cp.concatenate([used_mua[:3], used_mua[4:]]).T
        used_mua = cp.asnumpy(used_mua)
        dataset_output[:, :10] = np.concatenate([used_mus, used_mua], axis=1)
        np.save(os.path.join("result", result_folder, datasetpath,
                f"mus_{mus_run_idx}.npy"), dataset_output)
