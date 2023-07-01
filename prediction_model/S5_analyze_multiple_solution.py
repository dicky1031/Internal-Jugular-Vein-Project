# %%
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed


# %%
dataset_fileset = glob(os.path.join("dataset", "prediction_result", "test", "*.npy"))
dataset_fileset.sort(key=lambda x : int(x.split("\\")[-1].split("_")[0]))
for idx, dataset_file in enumerate(dataset_fileset):
    print(f'now processing {dataset_file}')
    if idx == 0:
        data = np.load(dataset_file)
    else:
        temp = np.load(dataset_file)
        data = np.concatenate((data, temp))

# %%
def get_similarity(idx1, spec1, spec1_SO2):
    topK = 10
    compare_data = data[np.where(data[:,41]!=spec1_SO2)]
    diff_spec = (spec1[:40] - compare_data[:, :40])/np.abs(compare_data[:, :40])*100
    similarity = np.mean(np.abs(diff_spec), axis=1)
    similarity_total_idx = np.argsort(similarity)
    similarity_topK_idx = similarity_total_idx[:topK]

    similarity_topK = similarity[similarity_topK_idx]
    spec_topK = compare_data[similarity_topK_idx]
    similarity_mean = np.mean(similarity)
    similarity_std = np.std(similarity)
    row = {'similarity_mean' : similarity_mean, 'similarity_std': similarity_std}
    for idx in range(spec1.shape[0]):
        row[f'spec1_value_{idx}'] = spec1[idx]
    for idx in range(similarity_topK.shape[0]):
        row[f'similarity_{idx}'] = similarity_topK[idx]
    for i in range(spec_topK.shape[0]):
        for j in range(spec_topK.shape[1]):
            row[f'similar_spec{i}_value_{j}'] = spec_topK[i,j]
    
    return row


data = data[~np.any(data[:,:40] == 0, axis=1)] # remove any zero reflectance
res = Parallel(n_jobs=-5)(delayed(get_similarity)(idx1, data[idx1], data[idx1,41]) for idx1 in tqdm(range(data[:100].shape[0])))
res = pd.DataFrame(res)
subject = 'ctchen'
result = 'prediction_model_formula2'
os.makedirs(os.path.join('result', subject, result), exist_ok=True)
res.to_csv(os.path.join('result', subject, result), index=False)



