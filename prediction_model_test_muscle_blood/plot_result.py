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

# %%
result_folder = "result1"

# %%
os.makedirs(os.path.join("pic", result_folder), exist_ok=True)
test_result = pd.read_csv(os.path.join("model_save", result_folder, "test.csv"))

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
muscle_SO2_dict = {}
for muscle_mua_change in test_result['muscle_mua_change'].unique():
    temp = test_result[test_result['muscle_mua_change']==muscle_mua_change]
    error = temp['error'].to_list()
    muscle_SO2_dict[f'{muscle_mua_change}_error'] = error
# do t-test one-tail
for muscle_mua_change in test_result['muscle_mua_change'].unique():
    statistic, pvalue = t_test(muscle_SO2_dict['0.0_error'],muscle_SO2_dict[f'{muscle_mua_change}_error'])
    muscle_SO2_dict[f'{muscle_mua_change}_pvalue'] = pvalue
    
data_y = []
data_x = []
p_val = []
for muscle_mua_change in sorted(test_result['muscle_mua_change'].unique(), reverse=True):
    data_y.append(muscle_SO2_dict[f'{muscle_mua_change}_error'])
    data_x.append(f'{int(muscle_mua_change)}%')
    p_val.append(muscle_SO2_dict[f'{muscle_mua_change}_pvalue'])

ax = sns.boxplot(data_y)
count = 1
for i in range(len(data_x)):
    if i==1:
        continue
    else:
        x1,x2 = 1,i
        y,h = max(test_result['error'])+count, 1
        #绘制横线位置
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="k") 
        #添加P值
        ax.text((x1+x2)*.5, y+h, convert_pvalue_to_asterisks(p_val[i]), ha='center', va='bottom', color="k")
        count += 15

plt.xticks([i for i in range(len(data_y))],data_x)
plt.xlabel("muscle SO2 change(%)")
plt.ylabel("error(prediction - true)")
plt.savefig(os.path.join("pic", result_folder, f"muscle_change_{muscle_mua_change}_boxplot.png"), dpi=300, format='png', bbox_inches='tight')
plt.show()


# %%
for muscle_mua_change in test_result['muscle_mua_change'].unique():
    temp = test_result[test_result['muscle_mua_change']==muscle_mua_change]
    output = temp['output']
    target = temp['target']
    error = temp['error']
    
    plt.plot(target,output, 'r.', markersize=5)
    plt.plot(target,target,'b')
    RMSE = np.sqrt(np.mean(np.square(error)))
    mean = np.mean(np.abs(error))
    std = np.std(np.abs(error))
    max_error = np.max(np.abs(error))
    plt.title(f"based on ijv_SO2=70% muscle_SO2_change:{muscle_mua_change}%\nmean error:{mean:.2f}% std:{std:.2f}% \nmax error:{max_error:.2f}% RMSE:{RMSE:.2f}%")
    plt.xlabel("truth $\u0394$SO2")
    plt.ylabel("predict $\u0394$SO2")
    plt.savefig(os.path.join("pic", result_folder, f"muscle_change_{muscle_mua_change}_RMSE.png"), dpi=300, format='png', bbox_inches='tight')
    plt.show()


