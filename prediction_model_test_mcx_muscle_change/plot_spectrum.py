# %%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import json
import sys
# Default settings
mpl.rcParams.update(mpl.rcParamsDefault)
os.chdir(sys.path[0])
plt.style.use("seaborn-darkgrid")

# %%
mus_types = ['high', 'medium', 'low']
mua_types = ['high', 'medium', 'low']
muscle_types = ['muscle_0', 'muscle_1', 'muscle_3', 'muscle_5', 'muscle_10']
# result_folder = "delta_OD"
short_SDS = 'SDS_1'
long_SDS = 'SDS_12'
subject = 'ctchen'
top_k = 10
for muscle_type in muscle_types:
    for mus_type in mus_types:
        for mua_type in mua_types:
            os.makedirs(os.path.join("pic", subject, "delta_OD", muscle_type, mus_type, mua_type, 'top_k_large_error'), exist_ok=True)
            os.makedirs(os.path.join("pic", subject, "delta_OD", muscle_type, mus_type, mua_type, 'top_k_small_error'), exist_ok=True)
            os.makedirs(os.path.join("pic", subject, "spectrum", muscle_type, mus_type, mua_type, 'top_k_small_error'), exist_ok=True)
            os.makedirs(os.path.join("pic", subject, "spectrum", muscle_type, mus_type, mua_type, 'top_k_small_error'), exist_ok=True)

with open(os.path.join('OPs_used', 'wavelength.json'), 'r') as f:
    wavelength = json.load(f)
    wavelength = wavelength['wavelength']

# %%
def plot_top_k_large_error_delta_OD(test_result, prediction_input, top_k,  muscle_type, mus_type, mua_type):
        top_k_error = abs(test_result['error_ijv_SO2']).nlargest(top_k)
        top_k_error_index = top_k_error.index
        for pic_id, error_idx in enumerate(top_k_error_index):
                error = test_result.iloc[error_idx]['error_ijv_SO2']
                used_id = test_result[test_result['error_ijv_SO2'] == error]['id']
                used_id = str(used_id)
                used_id = used_id.split()[1]
                dataset = prediction_input[prediction_input['id'] == used_id].to_numpy()
                dataset = dataset.reshape(-1)
                now_ijv_SO2 = dataset[41]*100
                now_muscle_SO2 = dataset[-1]*100
                plt.figure()
                plt.plot(wavelength, dataset[:20], label= '$\u0394$$IJV_{Large}$')
                plt.plot(wavelength, dataset[20:40], label= '$\u0394$$IJV_{Small}$')
                plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
                plt.xlabel("wavelength(nm)")
                plt.ylabel("$\u0394$OD")
                plt.title('$\u0394$OD @ $SO2_{ijv}$ :' + f' {now_ijv_SO2:.1f}%,' + '\n$SO2_{muscle}$ :' + f' {now_muscle_SO2:.1f}%, error : {error:.2f}%')
                plt.tight_layout()
                plt.savefig(os.path.join("pic", subject, "delta_OD", muscle_type, mus_type, mua_type, 'top_k_large_error',  f"{pic_id}_delta_OD.png"), dpi=300, format='png', bbox_inches='tight')
                plt.show()

# %%
def plot_top_k_small_error_delta_OD(test_result, prediction_input, top_k,  muscle_type, mus_type, mua_type):
        top_k_error = abs(test_result['error_ijv_SO2']).nsmallest(top_k)
        top_k_error_index = top_k_error.index
        for pic_id, error_idx in enumerate(top_k_error_index):
                error = test_result.iloc[error_idx]['error_ijv_SO2']
                used_id = test_result[test_result['error_ijv_SO2'] == error]['id']
                used_id = str(used_id)
                used_id = used_id.split()[1]
                dataset = prediction_input[prediction_input['id'] == used_id].to_numpy()
                dataset = dataset.reshape(-1)
                now_ijv_SO2 = dataset[41]*100
                now_muscle_SO2 = dataset[-1]*100
                plt.figure()
                plt.plot(wavelength, dataset[:20], label= '$\u0394$$IJV_{Large}$')
                plt.plot(wavelength, dataset[20:40], label= '$\u0394$$IJV_{Small}$')
                plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
                plt.xlabel("wavelength(nm)")
                plt.ylabel("$\u0394$OD")
                plt.title('$\u0394$OD @ $SO2_{ijv}$ :' + f' {now_ijv_SO2:.1f}%,' + '\n$SO2_{muscle}$ :' + f' {now_muscle_SO2:.1f}%, error : {error:.2f}%')
                plt.tight_layout()
                plt.savefig(os.path.join("pic", subject, "delta_OD", muscle_type, mus_type, mua_type, 'top_k_small_error',  f"{pic_id}_delta_OD.png"), dpi=300, format='png', bbox_inches='tight')
                plt.close()
                # plt.show()

# %%
def plot_top_k_large_error_spectrum(test_result, prediction_input, top_k,  muscle_type, mus_type, mua_type):
    top_k_error = abs(test_result['error_ijv_SO2']).nlargest(top_k)
    top_k_error_index = top_k_error.index
    for pic_id, error_idx in enumerate(top_k_error_index):
        error = test_result.iloc[error_idx]['error_ijv_SO2']
        used_id = test_result[test_result['error_ijv_SO2'] == error]['id']
        used_id = str(used_id)
        used_id = used_id.split()[1]
        dataset = prediction_input[prediction_input['id'] == used_id].to_numpy()
        dataset = dataset.reshape(-1)
        now_ijv_SO2 = dataset[41]*100
        now_muscle_SO2 = dataset[-1]*100
        large_T1_short_SDS = []
        large_T1_long_SDS = []
        large_T2_short_SDS = []
        large_T2_long_SDS = []
        small_T1_short_SDS = []
        small_T1_long_SDS = []
        small_T2_short_SDS = []
        small_T2_long_SDS = []
        
        based_ijv_SO2 = 0.7
        based_muscle_SO2 = 0.7
        used_blc = int(prediction_input[prediction_input['id'] == used_id]['blc'].iloc[0])
        used_ijv_SO2 = (prediction_input[prediction_input['id'] == used_id]['ijv_SO2_change'].iloc[0]) + based_ijv_SO2
        used_ijv_SO2 = f'{used_ijv_SO2:.2f}'
        used_ijv_SO2 = float(used_ijv_SO2)
        used_muscle_SO2 = (prediction_input[prediction_input['id'] == used_id]['muscle_SO2_change'].iloc[0]) + based_muscle_SO2
        used_muscle_SO2 = f'{used_muscle_SO2:.2f}'
        used_muscle_SO2 = float(used_muscle_SO2)
        for idx, wl in enumerate(wavelength):
            origin_dataset_large = pd.read_csv(os.path.join("dataset", subject, f"{subject}_dataset_large_{muscle_type}", f"{mus_type}", f"{wl}nm_mus_{idx+1}.csv"))
            origin_dataset_small = pd.read_csv(os.path.join("dataset", subject, f"{subject}_dataset_large_{muscle_type}", f"{mus_type}", f"{wl}nm_mus_{idx+1}.csv"))

            used_order = int(used_id.split('_')[-1])
            large_T1_short_SDS.append(origin_dataset_large[(origin_dataset_large['bloodConc']==used_blc) & (origin_dataset_large['used_SO2']==based_ijv_SO2) & (origin_dataset_large['muscle_SO2']==based_muscle_SO2)].iloc[used_order][short_SDS])
            large_T1_long_SDS.append(origin_dataset_large[(origin_dataset_large['bloodConc']==used_blc) & (origin_dataset_large['used_SO2']==based_ijv_SO2) & (origin_dataset_large['muscle_SO2']==based_muscle_SO2)].iloc[used_order][long_SDS])
            large_T2_short_SDS.append(origin_dataset_large[(origin_dataset_large['bloodConc']==used_blc) & (origin_dataset_large['used_SO2']==used_ijv_SO2) & (origin_dataset_large['muscle_SO2']==used_muscle_SO2)].iloc[used_order][short_SDS])
            large_T2_long_SDS.append(origin_dataset_large[(origin_dataset_large['bloodConc']==used_blc) & (origin_dataset_large['used_SO2']==used_ijv_SO2) & (origin_dataset_large['muscle_SO2']==used_muscle_SO2)].iloc[used_order][long_SDS])
            small_T1_short_SDS.append(origin_dataset_small[(origin_dataset_small['bloodConc']==used_blc) & (origin_dataset_small['used_SO2']==based_ijv_SO2) & (origin_dataset_small['muscle_SO2']==based_muscle_SO2)].iloc[used_order][short_SDS])
            small_T1_long_SDS.append(origin_dataset_small[(origin_dataset_small['bloodConc']==used_blc) & (origin_dataset_small['used_SO2']==based_ijv_SO2) & (origin_dataset_small['muscle_SO2']==based_muscle_SO2)].iloc[used_order][long_SDS])
            small_T2_short_SDS.append(origin_dataset_small[(origin_dataset_small['bloodConc']==used_blc) & (origin_dataset_small['used_SO2']==used_ijv_SO2) & (origin_dataset_small['muscle_SO2']==used_muscle_SO2)].iloc[used_order][short_SDS])
            small_T2_long_SDS.append(origin_dataset_small[(origin_dataset_small['bloodConc']==used_blc) & (origin_dataset_small['used_SO2']==used_ijv_SO2) & (origin_dataset_small['muscle_SO2']==used_muscle_SO2)].iloc[used_order][long_SDS])
        
        plt.figure()
        plt.plot(wavelength, large_T1_short_SDS, label='large_T1_short_SDS')
        plt.plot(wavelength, large_T1_long_SDS, label='large_T1_long_SDS')
        plt.plot(wavelength, large_T2_short_SDS, label='large_T2_short_SDS')
        plt.plot(wavelength, large_T2_long_SDS, label='large_T2_long_SDS')
        plt.plot(wavelength, small_T1_short_SDS, label='small_T1_short_SDS')
        plt.plot(wavelength, small_T1_long_SDS, label='small_T1_long_SDS')
        plt.plot(wavelength, small_T2_short_SDS, label='small_T2_short_SDS')
        plt.plot(wavelength, small_T2_long_SDS, label='small_T2_long_SDS')
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
        plt.xlabel("wavelength(nm)")
        plt.ylabel("Intensity")
        plt.title('spectrum @ $SO2_{ijv}$ :' + f' {now_ijv_SO2:.1f}%,' + '\n$SO2_{muscle}$ :' + f' {now_muscle_SO2:.1f}%, error : {error:.2f}%')
        plt.tight_layout()
        plt.savefig(os.path.join("pic", subject, "spectrum", muscle_type, mus_type, mua_type, 'top_k_large_error',  f"{pic_id}_delta_OD.png"), dpi=300, format='png', bbox_inches='tight')
        plt.close()
        # plt.show()

# %%
def plot_top_k_small_error_spectrum(test_result, prediction_input, top_k,  muscle_type, mus_type, mua_type):
    top_k_error = abs(test_result['error_ijv_SO2']).nsmallest(top_k)
    top_k_error_index = top_k_error.index
    for pic_id, error_idx in enumerate(top_k_error_index):
        error = test_result.iloc[error_idx]['error_ijv_SO2']
        used_id = test_result[test_result['error_ijv_SO2'] == error]['id']
        used_id = str(used_id)
        used_id = used_id.split()[1]
        dataset = prediction_input[prediction_input['id'] == used_id].to_numpy()
        dataset = dataset.reshape(-1)
        now_ijv_SO2 = dataset[41]*100
        now_muscle_SO2 = dataset[-1]*100
        large_T1_short_SDS = []
        large_T1_long_SDS = []
        large_T2_short_SDS = []
        large_T2_long_SDS = []
        small_T1_short_SDS = []
        small_T1_long_SDS = []
        small_T2_short_SDS = []
        small_T2_long_SDS = []
        
        based_ijv_SO2 = 0.7
        based_muscle_SO2 = 0.7
        used_blc = int(prediction_input[prediction_input['id'] == used_id]['blc'].iloc[0])
        used_ijv_SO2 = (prediction_input[prediction_input['id'] == used_id]['ijv_SO2_change'].iloc[0]) + based_ijv_SO2
        used_ijv_SO2 = f'{used_ijv_SO2:.2f}'
        used_ijv_SO2 = float(used_ijv_SO2)
        used_muscle_SO2 = (prediction_input[prediction_input['id'] == used_id]['muscle_SO2_change'].iloc[0]) + based_muscle_SO2
        used_muscle_SO2 = f'{used_muscle_SO2:.2f}'
        used_muscle_SO2 = float(used_muscle_SO2)
        for idx, wl in enumerate(wavelength):
            origin_dataset_large = pd.read_csv(os.path.join("dataset", subject, f"{subject}_dataset_large_{muscle_type}", f"{mus_type}", f"{wl}nm_mus_{idx+1}.csv"))
            origin_dataset_small = pd.read_csv(os.path.join("dataset", subject, f"{subject}_dataset_large_{muscle_type}", f"{mus_type}", f"{wl}nm_mus_{idx+1}.csv"))

            used_order = int(used_id.split('_')[-1])
            large_T1_short_SDS.append(origin_dataset_large[(origin_dataset_large['bloodConc']==used_blc) & (origin_dataset_large['used_SO2']==based_ijv_SO2) & (origin_dataset_large['muscle_SO2']==based_muscle_SO2)].iloc[used_order][short_SDS])
            large_T1_long_SDS.append(origin_dataset_large[(origin_dataset_large['bloodConc']==used_blc) & (origin_dataset_large['used_SO2']==based_ijv_SO2) & (origin_dataset_large['muscle_SO2']==based_muscle_SO2)].iloc[used_order][long_SDS])
            large_T2_short_SDS.append(origin_dataset_large[(origin_dataset_large['bloodConc']==used_blc) & (origin_dataset_large['used_SO2']==used_ijv_SO2) & (origin_dataset_large['muscle_SO2']==used_muscle_SO2)].iloc[used_order][short_SDS])
            large_T2_long_SDS.append(origin_dataset_large[(origin_dataset_large['bloodConc']==used_blc) & (origin_dataset_large['used_SO2']==used_ijv_SO2) & (origin_dataset_large['muscle_SO2']==used_muscle_SO2)].iloc[used_order][long_SDS])
            small_T1_short_SDS.append(origin_dataset_small[(origin_dataset_small['bloodConc']==used_blc) & (origin_dataset_small['used_SO2']==based_ijv_SO2) & (origin_dataset_small['muscle_SO2']==based_muscle_SO2)].iloc[used_order][short_SDS])
            small_T1_long_SDS.append(origin_dataset_small[(origin_dataset_small['bloodConc']==used_blc) & (origin_dataset_small['used_SO2']==based_ijv_SO2) & (origin_dataset_small['muscle_SO2']==based_muscle_SO2)].iloc[used_order][long_SDS])
            small_T2_short_SDS.append(origin_dataset_small[(origin_dataset_small['bloodConc']==used_blc) & (origin_dataset_small['used_SO2']==used_ijv_SO2) & (origin_dataset_small['muscle_SO2']==used_muscle_SO2)].iloc[used_order][short_SDS])
            small_T2_long_SDS.append(origin_dataset_small[(origin_dataset_small['bloodConc']==used_blc) & (origin_dataset_small['used_SO2']==used_ijv_SO2) & (origin_dataset_small['muscle_SO2']==used_muscle_SO2)].iloc[used_order][long_SDS])
        
        plt.figure()
        plt.plot(wavelength, large_T1_short_SDS, label='large_T1_short_SDS')
        plt.plot(wavelength, large_T1_long_SDS, label='large_T1_long_SDS')
        plt.plot(wavelength, large_T2_short_SDS, label='large_T2_short_SDS')
        plt.plot(wavelength, large_T2_long_SDS, label='large_T2_long_SDS')
        plt.plot(wavelength, small_T1_short_SDS, label='small_T1_short_SDS')
        plt.plot(wavelength, small_T1_long_SDS, label='small_T1_long_SDS')
        plt.plot(wavelength, small_T2_short_SDS, label='small_T2_short_SDS')
        plt.plot(wavelength, small_T2_long_SDS, label='small_T2_long_SDS')
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                        fancybox=True, shadow=True)
        plt.xlabel("wavelength(nm)")
        plt.ylabel("Intensity")
        plt.title('spectrum @ $SO2_{ijv}$ :' + f' {now_ijv_SO2:.1f}%,' + '\n$SO2_{muscle}$ :' + f' {now_muscle_SO2:.1f}%, error : {error:.2f}%')
        plt.tight_layout()
        plt.savefig(os.path.join("pic", subject, "spectrum", muscle_type, mus_type, mua_type, 'top_k_small_error',  f"{pic_id}_delta_OD.png"), dpi=300, format='png', bbox_inches='tight')
        plt.close()
        # plt.show()

if __name__ == "__main__":
    # %%
    for muscle_type in muscle_types:
        for mus_type in mus_types:
            for mua_type in mua_types:
                test_result = pd.read_csv(os.path.join("model_test", subject, f"{mus_type}_scatter_prediction_input_{muscle_type}", f"{mua_type}_absorption", "test.csv"))
                prediction_input = pd.read_csv(os.path.join("dataset", subject, f"{mus_type}_scatter_prediction_input_{muscle_type}", f"{mua_type}_absorption", "prediction_input.csv"))
                # plot_top_k_large_error_delta_OD(test_result, prediction_input, top_k,  muscle_type, mus_type, mua_type)
                # plot_top_k_small_error_delta_OD(test_result, prediction_input, top_k,  muscle_type, mus_type, mua_type)
                plot_top_k_large_error_spectrum(test_result, prediction_input, top_k,  muscle_type, mus_type, mua_type)
                plot_top_k_small_error_spectrum(test_result, prediction_input, top_k,  muscle_type, mus_type, mua_type)
            
                    


