# %%
import matplotlib.pyplot as plt
from ANN_models import PredictionModel
import os 
import json
import torch
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
import sys
import matplotlib as mpl
# Default settings
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("seaborn-darkgrid")
os.chdir(sys.path[0])

# %%
def t_test(group1, group2):
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1)
    std2 = np.std(group2)
    nobs1 = len(group1)
    nobs2 = len(group2)
    
    modified_std1 = np.sqrt(np.float32(nobs1)/
                    np.float32(nobs1-1)) * std1
    modified_std2 = np.sqrt(np.float32(nobs2)/
                    np.float32(nobs2-1)) * std2
    
    statistic, pvalue = stats.ttest_ind_from_stats( 
            mean1=mean1, std1=modified_std1, nobs1=nobs1,   
            mean2=mean2, std2=modified_std2, nobs2=nobs2)

    return statistic, pvalue

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

# %%
mus_types = ['low', 'high', 'medium']
mua_types = ['all', 'low', 'high', 'medium']
muscle_types = ['muscle_0', 'muscle_1', 'muscle_3', 'muscle_5', 'muscle_10']
subject = 'ctchen'
result = 'surrogate_formula2'

for muscle_type in muscle_types:
    for mus_type in mus_types:
        for mua_type in mua_types:
            
            result_folder = os.path.join(f"{mus_type}_scatter_prediction_input_{muscle_type}", f"{mua_type}_absorption")
            # %%
            os.makedirs(os.path.join("pic", subject, result, result_folder), exist_ok=True)
            test_result = pd.read_csv(os.path.join("model_test", subject, result, result_folder, "test.csv"))

            # %%
            muscle_SO2_dict = {}
            for muscle_mua_change in test_result['muscle_mua_change'].unique():
                temp = test_result[test_result['muscle_mua_change']==muscle_mua_change]
                error = temp['error_ijv_SO2'].to_list()
                muscle_SO2_dict[f'{muscle_mua_change}_error_ijv_SO2'] = error
            # do t-test one-tail
            for muscle_mua_change in test_result['muscle_mua_change'].unique():
                statistic, pvalue = t_test(muscle_SO2_dict['0.0_error_ijv_SO2'],muscle_SO2_dict[f'{muscle_mua_change}_error_ijv_SO2'])
                muscle_SO2_dict[f'{muscle_mua_change}_pvalue'] = pvalue
                
            data_y = []
            data_x = []
            p_val = []
            for muscle_mua_change in sorted(test_result['muscle_mua_change'].unique(), reverse=True):
                data_y.append(muscle_SO2_dict[f'{muscle_mua_change}_error_ijv_SO2'])
                data_x.append(f'{muscle_mua_change:.1f}%')
                p_val.append(muscle_SO2_dict[f'{muscle_mua_change}_pvalue'])

            ax = sns.boxplot(data=data_y)
            count = 5
            for i in range(len(data_x)):
                if i==5:
                    continue
                else:
                    x1,x2 = 5,i
                    y,h = max(test_result['error_ijv_SO2'])+count, 1
                    #绘制横线位置
                    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="k") 
                    #添加P值
                    ax.text((x1+x2)*.5, y+h, convert_pvalue_to_asterisks(p_val[i]), ha='center', va='bottom', color="k")
                    count += 15
            plt.figure()
            plt.xticks([i for i in range(len(data_y))],data_x)
            plt.xlabel("muscle SO2 change(%)")
            plt.ylabel("error(prediction - true)")
            plt.savefig(os.path.join("pic", subject, result, result_folder, f"muscle_change_{muscle_mua_change}_boxplot.png"), dpi=300, format='png', bbox_inches='tight')
            plt.close()
            # plt.show()


            # %%
            for muscle_mua_change in test_result['muscle_mua_change'].unique():
                temp = test_result[test_result['muscle_mua_change']==muscle_mua_change]
                output = temp['output_ijv_SO2']
                target = temp['target_ijv_SO2']
                error = temp['error_ijv_SO2']
                plt.figure()
                plt.plot(target,output, 'r.', markersize=5, label='Predict')
                plt.plot(target,target,'b', label='Truth')
                RMSE = np.sqrt(np.mean(np.square(error)))
                mean = np.mean(np.abs(error))
                std = np.std(np.abs(error))
                max_error = np.max(np.abs(error))
                plt.title(f"based on ijv_SO2=70% muscle_SO2_change:{muscle_mua_change}%\nmean error:{mean:.2f}% std:{std:.2f}% \nmax error:{max_error:.2f}% RMSE:{RMSE:.2f}%")
                plt.xlabel("truth $\u0394$SO2")
                plt.ylabel("predict $\u0394$SO2")
                plt.legend(loc=(1.01,0.8))
                plt.savefig(os.path.join("pic", subject, result, result_folder, f"muscle_change_{muscle_mua_change}_RMSE_ijv.png"), dpi=300, format='png', bbox_inches='tight')
                plt.close()
                # plt.show()


