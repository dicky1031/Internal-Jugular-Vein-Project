from mcx_ultrasound_opsbased import MCX
import calculateR_CV
import json
import os
import numpy as np
import pandas as pd
from glob import glob
from time import sleep
from tqdm import tqdm
import time
import sys
# %% move to current file path
os.chdir(sys.path[0])


# %% run

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def run_mcx(result_folder, subject, mus_start, mus_end, NA_enable, NA, runningNum, cvThreshold, repeatTimes, ijv_type):
    mus_set = np.load(os.path.join("OPs_used", "mus_set.npy"))
    timer = Timer()
    ID = f'{subject}_ijv_{ijv_type}'
    for run_idx in tqdm(range(mus_start, mus_end+1)):
        now = time.time()
        #  Setting
        session = f"run_{run_idx}"
        sessionID = os.path.join("dataset", result_folder, ID, session)

        #  load mua for calculating reflectance
        with open(os.path.join(sessionID, "mua_test.json")) as f:
            mua = json.load(f)
        muaUsed = [mua["1: Air"],
                   mua["2: PLA"],
                   mua["3: Prism"],
                   mua["4: Skin"],
                   mua["5: Fat"],
                   mua["6: Muscle"],
                   mua["7: Muscle or IJV (Perturbed Region)"],
                   mua["8: IJV"],
                   mua["9: CCA"]
                   ]

        #  Do simulation
        # initialize
        simulator = MCX(sessionID)

        with open(os.path.join(sessionID, "config.json")) as f:
            config = json.load(f)
        simulationResultPath = os.path.join(
            config["OutputPath"], session, "post_analysis", f"{session}_simulation_result.json")

        with open(simulationResultPath) as f:
            simulationResult = json.load(f)
        existedOutputNum = simulationResult["RawSampleNum"]
        # run forward mcx
        if runningNum:
            for idx in range(existedOutputNum, existedOutputNum+runningNum):
                # run
                simulator.run(idx)
                if NA_enable:
                    simulator.NA_adjust(NA)
                # save progress
                simulationResult["RawSampleNum"] = idx+1
                with open(simulationResultPath, "w") as f:
                    json.dump(simulationResult, f, indent=4)
            mean, CV = calculateR_CV.calculate_R_CV(
                sessionID, session, "mua_test.json")
            print(
                f"Session name: {sessionID} \n Reflectance mean: {mean} \nCV: {CV} ", end="\n\n")
            # remove file
            # remove_list = glob(os.path.join(
            #     config["OutputPath"], session, "mcx_output", "*.jdat"))
            # remove_list.sort(key=lambda x: int(x.split("_")[-2]))
            # remove_list = remove_list[1:]
            # for idx in range(len(remove_list)):
            #     os.remove(remove_list[idx])

        else:
            # run stage1 : run N sims to precalculate CV
            for idx in range(repeatTimes):
                # run
                simulator.run(idx)
                if NA_enable:
                    simulator.NA_adjust(NA)
                # save progress
                simulationResult["RawSampleNum"] = idx+1
                with open(simulationResultPath, "w") as f:
                    json.dump(simulationResult, f, indent=4)
            # calculate reflectance
            mean, CV = calculateR_CV.calculate_R_CV(
                sessionID, session, "mua_test.json")
            print(
                f"Session name: {sessionID} \n Reflectance mean: {mean} \nCV: {CV} \n Predict CV: {CV/np.sqrt(repeatTimes)}", end="\n\n")
            # run stage2 : run more sim to make up cvThreshold
            # reflectanceCV = {k: simulationResult["GroupingSampleCV"][k]
            #                  for k in simulationResult["GroupingSampleCV"]}

            predict_CV = max(CV)/np.sqrt(repeatTimes)
            while (predict_CV > cvThreshold):
                needAddOutputNum = int(
                    np.ceil((max(CV)**2)/(cvThreshold**2)) - repeatTimes)
                if needAddOutputNum > 0:
                    for idx in range(repeatTimes, repeatTimes+needAddOutputNum):
                        # run
                        simulator.run(idx)
                        if NA_enable:
                            simulator.NA_adjust(NA)
                        # save progress
                        simulationResult["RawSampleNum"] = idx+1
                        with open(simulationResultPath, "w") as f:
                            json.dump(simulationResult, f, indent=4)
                    # calculate reflectance
                    mean, CV = calculateR_CV.calculate_R_CV(
                        sessionID, session, "mua_test.json")
                    print(
                        f"Session name: {sessionID} \n Reflectance mean: {mean} \nCV: {CV} \n Predict CV: {CV/np.sqrt(repeatTimes+needAddOutputNum)}", end="\n\n")
                    predict_CV = max(CV)/np.sqrt(repeatTimes+needAddOutputNum)
                    repeatTimes = repeatTimes + needAddOutputNum
                    
            with open(simulationResultPath) as f:
                simulationResult = json.load(f)
            simulationResult['elapsed time'] = time.time() - now
            with open(simulationResultPath, "w") as f:
                json.dump(simulationResult, f, indent=4)

        print('ETA:{}/{}'.format(timer.measure(),
                                 timer.measure(run_idx / mus_set.shape[0])))
        sleep(0.01)


if __name__ == "__main__":

    # script setting
    # photon 1e9 take 1TB  CV 0.29%~0.81%  13mins per mus
    # photon 3e8 take 350GB CV 0.48%~1.08% 4mins per mus  wmc 110 mins per mus
    # -------------------------------------
    # photon 3e8 take 7mins for 10sims 24MB per file
    # photon 1e9 82MB per file 23mins

    # ID = sys.argv[1] #ID = "ctchen_ijv_small_to_large"
    # mus_start = int(sys.argv[2])
    # mus_end = int(sys.argv[3])

    result_folder = "kb"
    subject = "kb"
    ijv_type = 'small_to_large'
    mus_start = 1103
    mus_end = 1225
    NA_enable = 1  # 0 no NA, 1 consider NA
    NA = 0.22
    runningNum = 0  # (Integer or False)self.session
    cvThreshold = 3
    repeatTimes = 10
    run_mcx(result_folder, subject, mus_start, mus_end, NA_enable,
            NA, runningNum, cvThreshold, repeatTimes, ijv_type)
