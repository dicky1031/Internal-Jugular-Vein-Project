# %%
import os
import shutil
import json
import numpy as np
import sys

# %% move to current file path
os.chdir(sys.path[0])
#%%
result_folder = "ctchen"
subject = "ctchen"
PhotonNum = 1e7
# %% run
ijv_types = ['large_to_small', 'small_to_large']
mus_types = ['high', 'medium', 'low']


# copy config.json ijv_dense_symmetric_detectors_backgroundfiber_pmc.json model_parameters.json mua_test.json to each sim
copylist = ["config.json",
            "ijv_dense_symmetric_detectors_backgroundfiber_pmc.json",
            "model_parameters.json",
            "mua_test.json"]

for ijv_type in ijv_types:
    sessionID = f'{subject}_ijv_{ijv_type}'

    os.makedirs(os.path.join(
        "dataset", result_folder, sessionID), exist_ok=True)
    
    for mus_type in mus_types:
        mus_set = np.load(os.path.join("OPs_used", f"{mus_type}_mus_set.npy"))
        os.makedirs(os.path.join(
        "dataset", result_folder, sessionID, mus_type), exist_ok=True)
        
        # create runfile folder
        for run_idx in range(1, mus_set.shape[0]+1):
            run_name = f"run_{run_idx}"
            os.makedirs(os.path.join("dataset", result_folder,
                        sessionID, mus_type, run_name), exist_ok=True)
            for filename in copylist:
                src = os.path.join("input_template", filename)
                dst = os.path.join("dataset", result_folder,
                                sessionID, mus_type, run_name, filename)
                shutil.copyfile(src, dst)

                if filename == "config.json":
                    with open(dst) as f:
                        config = json.load(f)
                    config["SessionID"] = run_name
                    config["PhotonNum"] = PhotonNum
                    config["BinaryPath"] = os.path.join(os.getcwd(), "bin")
                    config["VolumePath"] = os.path.join(os.getcwd(
                    ), "ultrasound_image_processing", f"{subject}_perturbed_small_to_large.npy")
                    config["MCXInputPath"] = os.path.join(os.getcwd(
                    ), "dataset", result_folder, sessionID, mus_type,  run_name, "ijv_dense_symmetric_detectors_backgroundfiber_pmc.json")
                    config["OutputPath"] = os.path.join(
                        os.getcwd(), "dataset", result_folder, sessionID, mus_type)
                    config["Type"] = sessionID
                    with open(dst, "w") as f:
                        json.dump(config, f, indent=4)

                if filename == "ijv_dense_symmetric_detectors_backgroundfiber_pmc.json":
                    with open(dst) as f:
                        mcxInput = json.load(f)
                    # 0 : Fiber
                    # 1 : Air
                    # 2 : PLA
                    # 3 : Prism
                    # 4 : Skin
                    mcxInput["Domain"]["Media"][4]["mus"] = mus_set[run_idx-1][0]
                    # 5 : Fat
                    mcxInput["Domain"]["Media"][5]["mus"] = mus_set[run_idx-1][1]
                    # 6 : Muscle
                    mcxInput["Domain"]["Media"][6]["mus"] = mus_set[run_idx-1][2]
                    # 7 : Muscle or IJV (Perturbed Region)
                    if sessionID.find("small_to_large") != -1:
                        # muscle
                        mcxInput["Domain"]["Media"][7]["mus"] = mus_set[run_idx-1][2]
                    elif sessionID.find("large_to_small") != -1:
                        # ijv
                        mcxInput["Domain"]["Media"][7]["mus"] = mus_set[run_idx-1][3]
                    else:
                        raise Exception(
                            "Something wrong in your config[VolumePath] !")
                    # 8 : IJV
                    mcxInput["Domain"]["Media"][8]["mus"] = mus_set[run_idx-1][3]
                    # 9 : CCA
                    mcxInput["Domain"]["Media"][9]["mus"] = mus_set[run_idx-1][4]
                    
                    
                    
                    with open(dst, "w") as f:
                        json.dump(mcxInput, f, indent=4)

                if filename == "model_parameters.json":
                    with open(dst) as f:
                        modelParameters = json.load(f)
                    modelParameters["OptParam"]["Skin"]["mus"] = mus_set[run_idx-1][0]
                    modelParameters["OptParam"]["Fat"]["mus"] = mus_set[run_idx-1][1]
                    modelParameters["OptParam"]["Muscle"]["mus"] = mus_set[run_idx-1][2]
                    modelParameters["OptParam"]["IJV"]["mus"] = mus_set[run_idx-1][3]
                    modelParameters["OptParam"]["CCA"]["mus"] = mus_set[run_idx-1][4]
                    modelParameters['HardwareParam']['Source']['Beam']['ProfilePath'] = os.path.join(
                        "input_template", "shared_files", "model_input_related")
                    with open(dst, "w") as f:
                        json.dump(modelParameters, f, indent=4)


