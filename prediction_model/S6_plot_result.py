import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import json
import time

# import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300
#%% data preprocessing
class dataload(Dataset):
    def __init__(self, root):
        xy = np.load(root)
        mus_set_path = "mus_set.npy"
        mua_set_path = "mua_set.npy"
        mus_set = np.load(mus_set_path)
        mua_set = np.load(mua_set_path)
        
        max_mus = np.repeat(np.max(mus_set, axis=0)[:3],20)
        max_mua = np.repeat(np.max(mua_set, axis=0)[:3],20)
        bloodConc_max = np.array([174])
        x_max = torch.from_numpy(np.concatenate((max_mus,max_mua,bloodConc_max)))
        
        min_mus = np.repeat(np.min(mus_set, axis=0)[:3],20)
        min_mua = np.repeat(np.min(mua_set, axis=0)[:3],20)
        bloodConc_min = np.array([138])
        x_min = torch.from_numpy(np.concatenate((min_mus,min_mua,bloodConc_min)))
        
        idx = [i for i in range(282,302)] + [i for i in range(322,342)] + [i for i in range(362,382)] + [i for i in range(402,422)] + [i for i in range(41,101)] + [i for i in range(141,201)] + [241] 
        # x = xy[:,idx]
        # x[:,:40] = x[:,:40]*10**5
        # x[:,40:201] = (x[:,40:201] - x_min.numpy()) / (x_max.numpy() - x_min.numpy())
        # x[:,201:221] = x[:,201:221]*10**8
        # x[:,241:261] = x[:,241:261]*10**8
        # x[:,281:301] = x[:,281:301]*10**8
        # x[:,321:341] = x[:,321:341]*10**8
        
        
        # idx = [i for i in range(282,302)] + [i for i in range(322,342)] + [i for i in range(362,382)] + [i for i in range(402,422)]
        # self.x = torch.from_numpy(xy[:,262:422])
        # self.x[:,0:20] =  self.x[:,0:20]*10**8
        # self.x[:,40:60] = self.x[:,40:60]*10**8
        # self.x[:,80:100] = self.x[:,80:100]*10**8
        # self.x[:,120:140] = self.x[:,120:140]*10**8
        
        self.x = torch.from_numpy(xy[:,idx]) # small value :0, 10^-10
        # self.x[:,0:160] = self.x[:,0:160]*10**8
        # self.x[:,0:20] =  self.x[:,0:20]*10**8
        # self.x[:,40:60] = self.x[:,40:60]*10**8
        # self.x[:,80:100] = self.x[:,80:100]*10**8
        # self.x[:,120:140] = self.x[:,120:140]*10**8
        
        self.x[:,80:201] =  (self.x[:,80:201] - x_min) / (x_max - x_min)
        # self.x[:,201:221] = self.x[:,201:221]*10**8
        # self.x[:,241:261] = self.x[:,241:261]*10**8
        # self.x[:,281:301] = self.x[:,281:301]*10**8
        # self.x[:,321:341] = self.x[:,321:341]*10**8
        self.y = torch.from_numpy(xy[:,40]) 
        self.parameters = torch.from_numpy(xy[:,41:])
        self.n_samples = xy.shape[0]
                
    def __getitem__(self, index):
        
        return self.x[index], self.y[index], self.parameters[index]
        
    def __len__(self):
        
        return self.n_samples

def data_preprocess(dataset, batch_size, test_split, shuffle_dataset, random_seed):
    # create data indice for training and testing splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # count out split size
    split = int(np.floor(test_split*dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:],indices[:split]

    # creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, test_loader

#%% model1
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(201, 256),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
        
    def forward(self, x):
        return self.net(x)
    
#%% plot pred
def pred():
    if not os.path.isdir(os.path.join("pic",f"{local_time}")):
        os.mkdir(os.path.join("pic",f"{local_time}"))
    
    
    model.eval()
    error = []
    error_dict = {}

    max_error = 0
    SO2 = [(i-70)/100 for i in range(40,91,1)]
    # SO2.pop(SO2.index(0.0))
    # SO2_text = [f'{(i-70)}%' for i in range(40,91,1)]
    for s in SO2:
        error_dict[str(round(100*s))] = []
    
    true_SO2 = []
    pred_SO2 = []
    error_SO2 = []
    for batch_idx, (data,target, parameters) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output.view(-1)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        # y = torch.exp(-output).detach().cpu().numpy()
        # x = torch.exp(-target).detach().cpu().numpy()
        error += list(100*(torch.abs((torch.tensor(output)-torch.tensor(target)))).numpy())
        e = torch.abs((torch.tensor(output)-torch.tensor(target))).max().item()
        if e*100 > max_error:
            max_error = e*100
        # error += torch.sqrt(torch.square((torch.tensor(output)-torch.tensor(target))).mean()).item()
        for i in range(target.shape[0]):
            error_dict[str(round(100*target[i]))].append(np.square(100*output[i]-100*target[i]))
            true_SO2.append(round(100*target[i]))
            pred_SO2.append(100*output[i])
            error_SO2.append(100*(output[i]-target[i]))
            
        plt.plot(target*100,output*100, 'r.', markersize=5)
        plt.plot(target*100,target*100,'b')
    
    error = np.array(error)
    RMSE = np.sqrt(np.mean(np.square(error)))
    std = np.std(error)
    plt.title(f"based on SO2=70% \nmean error:{np.mean(error):.2f}% std:{std:.2f}% \n RMSE:{RMSE:.2f}%")
    # plt.xticks(SO2, SO2_text)
    # plt.yticks(SO2, SO2_text)
    plt.xlabel("truth $\u0394$SO2")
    plt.ylabel("predict $\u0394$SO2")
    plt.legend(["predict", "optimal"])
    plt.tight_layout()
    plt.savefig(os.path.join("pic",f"{local_time}","RMSPE.png"))
    plt.show()
    
    ## plot spectrum
    sort_error = sorted(error)
    outlier_parameters = []
    accurate_parameterss = []
    for batch_idx, (data,target, parameters) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        output = model(data)
        output = output.view(-1)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        batch_error = 100*(torch.abs((torch.tensor(output)-torch.tensor(target)))).numpy()
        for e_idx, e in enumerate(batch_error):
            if e > sort_error[-10]:
                outlier_parameters.append((parameters[e_idx],e,target[e_idx]))
            if e < sort_error[10]:
                accurate_parameterss.append((parameters[e_idx],e,target[e_idx]))
    plot_spectrum(out_acc="outlier", analyze_parameters=outlier_parameters)
    plot_spectrum(out_acc="accurate", analyze_parameters=accurate_parameterss)
    ###########################
    
    # find_len = []
    # for k in error_dict.keys():
    #     find_len.append(len(error_dict[k]))
    # find_len = min(find_len)
    # for k in error_dict.keys():
    #     error_dict[k] = error_dict[k][:find_len]
    
    #############################
    plot_y = [] # RMSE 
    plot_y1 = [] # std of RMSE
    plot_y2 = []
    plot_x = [] # delta SO2
    for k in error_dict.keys():
       plot_y.append(np.sqrt(np.mean(error_dict[k])))
       plot_y1.append(np.sqrt(np.mean(error_dict[k])) + np.sqrt(np.std(error_dict[k])))
       plot_y2.append(np.sqrt(np.mean(error_dict[k])) - np.sqrt(np.std(error_dict[k])))
       plot_x.append(int(k))
    plt.fill_between(plot_x,plot_y1,plot_y2, alpha=0.5, label="\u03BC \u00B1 \u03C3")
    plt.title("Prediction Error for Each $\u0394$SO2")
    plt.plot(plot_x,plot_y, label="\u03BC")
    plt.legend()
    plt.ylim([0, max(max(plot_y1,plot_y2))])
    plt.xlabel("$\u0394$SO2(%)")
    plt.ylabel("RMSE (%)")
    plt.tight_layout()
    plt.savefig(os.path.join("pic",f"{local_time}","RMSE.png"))
    plt.show()
        
    
    
    df = pd.DataFrame({"true_SO2(%)" : true_SO2,
                     "pred_SO2(%)" : pred_SO2,
                     "error_SO2(%)" : error_SO2})
    
    plt.figure(figsize=(8,6))
    sns.boxplot(x="true_SO2(%)", y="error_SO2(%)", data=df)
    plt.tight_layout()
    plt.savefig(os.path.join("pic",f"{local_time}","boxplot.png"))
    plt.show()
    df['error_SO2(%)'].hist()
    plt.xlabel("error(%)")
    plt.ylabel("accumulate data")
    plt.tight_layout()
    plt.savefig(os.path.join("pic",f"{local_time}","histplot.png"))
    plt.show()
    # 填充效果查看
    boxplot_fill(df['error_SO2(%)']).hist()
    plt.xlabel("error(%)")
    plt.ylabel("accumulate data")
    plt.tight_layout()
    plt.savefig(os.path.join("pic",f"{local_time}","histplot_remove_outlier.png"))
    plt.show()
    # 进行赋值
    plt.figure(figsize=(8,6))
    df['error_SO2(%)'] = boxplot_fill(df['error_SO2(%)'])
    sns.boxplot(x="true_SO2(%)", y="error_SO2(%)", data=df)
    plt.tight_layout()
    plt.savefig(os.path.join("pic",f"{local_time}","boxplot_remove_outlier.png"))
    plt.show()
    
def boxplot_fill(col):
    
     # 计算iqr：数据四分之三分位值与四分之一分位值的差
     iqr = col.quantile(0.75)-col.quantile(0.25)
     # 根据iqr计算异常值判断阈值
     u_th = col.quantile(0.75) + 1.5*iqr # 上界
     l_th = col.quantile(0.25) - 1.5*iqr # 下界
     # 定义转换函数：如果数字大于上界则用上界值填充，小于下界则用下界值填充。
     def box_trans(x):
      if x > u_th:
       return u_th
      elif x < l_th:
       return l_th
      else:
       return x
     return col.map(box_trans)
 
def plot_spectrum(out_acc, analyze_parameters):
    if not os.path.isdir(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}")):
        os.mkdir(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}"))
    
    used_wl = list(np.rint(np.linspace(700, 900, 20)).astype(int))
    for idx, (parameters,e,ground_truth) in enumerate(analyze_parameters):
        if not os.path.isdir(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}")):
            os.mkdir(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}"))
        # plot spectrum
        skin_mus = parameters.numpy()[:20]
        fat_mus = parameters.numpy()[20:40]
        muscle_mus = parameters.numpy()[40:60]
        IJV_mus = parameters.numpy()[60:80]
        CCA_mus = parameters.numpy()[80:100]
        skin_mua = parameters.numpy()[100:120]
        fat_mua = parameters.numpy()[120:140]
        muscle_mua = parameters.numpy()[140:160]
        IJV_mua = parameters.numpy()[160:180]
        CCA_mua = parameters.numpy()[180:200]
        bloodConc = parameters.numpy()[200]
        IJV_70_mua = parameters.numpy()[201:221]
        T2_large = parameters.numpy()[221:261]
        T2_small = parameters.numpy()[261:301]
        T1_large = parameters.numpy()[301:341]
        T1_small = parameters.numpy()[341:381]
        tissue = ["skin mus", "fat mus", "muscle mus", "IJV mus", "CCA mus",
                  "skin mua", "fat mua", "muscle mua", "IJV mua", "CCA mua"]
        
        for t_idx, t in enumerate(tissue):
            if t.split()[-1] == "mus":
                if t.split()[0] == "CCA" or t.split()[0] == "IJV":
                    plot_tissue = "blood"
                else:
                    plot_tissue = t.split()[0]
                tissue_bound_max = mus_bound[plot_tissue][0]
                tissue_bound_min = mus_bound[plot_tissue][-1]
            elif t.split()[-1] == "mua":
                if t.split()[0] == "CCA" or t.split()[0] == "IJV":
                    plot_tissue = t.split()[0].lower()
                else:
                    plot_tissue = t.split()[0]
                tissue_bound_max = mua_bound[plot_tissue][0]
                tissue_bound_min = mua_bound[plot_tissue][-1]
            else:
                raise Exception("error tissue")
            
            if plot_tissue == "ijv":
                used_spectrum = parameters.numpy()[20*t_idx:20*(t_idx+1)]
                used_spectrum = (used_spectrum - tissue_bound_min)/ (tissue_bound_max - tissue_bound_min)
                IJV_70_mua_used_spec = (IJV_70_mua - tissue_bound_min)/ (tissue_bound_max - tissue_bound_min)
                plt.plot(used_wl, 100*used_spectrum, label=f"{t}")
                plt.plot(used_wl, 100*IJV_70_mua_used_spec, label=f"{t}")
            else:
                used_spectrum = parameters.numpy()[20*t_idx:20*(t_idx+1)]
                used_spectrum = (used_spectrum - tissue_bound_min)/ (tissue_bound_max - tissue_bound_min)
                plt.plot(used_wl, 100*used_spectrum, label=f"{t}")
            plt.title(f"{t} \n error:{e:.2f}%")
            plt.xlabel("wavelength (nm)")
            plt.ylabel("parameter range(%))")
            plt.legend(loc="upper left", bbox_to_anchor=(1.05,1.0))
            plt.tight_layout()
            plt.savefig(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}",f"{t}_used_range.png"),bbox_inches="tight")
            plt.show()
        
        for t_idx, t in enumerate(tissue):
            if t.split()[-1] == "mus":
                if t.split()[0] == "CCA" or t.split()[0] == "IJV":
                    plot_tissue = "blood"
                else:
                    plot_tissue = t.split()[0]
                tissue_bound_max = mus_bound[plot_tissue][0]
                tissue_bound_min = mus_bound[plot_tissue][-1]
            elif t.split()[-1] == "mua":
                if t.split()[0] == "CCA" or t.split()[0] == "IJV":
                    plot_tissue = t.split()[0].lower()
                else:
                    plot_tissue = t.split()[0]
                tissue_bound_max = mua_bound[plot_tissue][0]
                tissue_bound_min = mua_bound[plot_tissue][-1]
            else:
                raise Exception("error tissue")

            used_spectrum = parameters.numpy()[20*t_idx:20*(t_idx+1)]
            used_spectrum = (used_spectrum - tissue_bound_min)/ (tissue_bound_max - tissue_bound_min)
            plt.plot(used_wl, 100*used_spectrum, label=f"{t}")
        plt.title(f"optical parameter used range \n error:{e:.2f}%")
        plt.xlabel("wavelength (nm)")
        plt.ylabel("parameter range(%))")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05,1.0))
        plt.tight_layout()
        plt.savefig(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}","merge_used_range.png"),bbox_inches="tight")
        plt.show()
        
        
        for t_idx, t in enumerate(tissue):
            if t.split()[-1] == "mus":
                if t.split()[0] == "CCA" or t.split()[0] == "IJV":
                    plot_tissue = "blood"
                else:
                    plot_tissue = t.split()[0]
                tissue_bound_max = mus_bound[plot_tissue][0]
                tissue_bound_min = mus_bound[plot_tissue][-1]
            elif t.split()[-1] == "mua":
                if t.split()[0] == "CCA" or t.split()[0] == "IJV":
                    plot_tissue = t.split()[0].lower()
                else:
                    plot_tissue = t.split()[0]
                tissue_bound_max = mua_bound[plot_tissue][0]
                tissue_bound_min = mua_bound[plot_tissue][-1]
            else:
                raise Exception("error tissue")
            
            used_spectrum = parameters.numpy()[20*t_idx:20*(t_idx+1)]
            used_spectrum = (used_spectrum - tissue_bound_min)/ (tissue_bound_max - tissue_bound_min)
            if t != "IJV mua":
                plt.plot(used_wl, 100*used_spectrum, label=f"{t}")
        plt.title(f"optical parameter used range (exceept IJV mua) \n error:{e:.2f}%")
        plt.xlabel("wavelength (nm)")
        plt.ylabel("parameter range(%))")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05,1.0))
        plt.tight_layout()
        plt.savefig(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}","merge_used_range_except_ijv.png"),bbox_inches="tight")
        plt.show()
            
        for t_idx, t in enumerate(tissue):
            if t == "IJV mua":
                plt.plot(used_wl, IJV_70_mua, label="IJV  mua baseline (70%)")
                plt.plot(used_wl, parameters.numpy()[20*t_idx:20*(t_idx+1)], label=f"{t}")
                plt.legend()
            else:
                plt.plot(used_wl, parameters.numpy()[20*t_idx:20*(t_idx+1)], label=f"{t}")
            plt.title(f"{t}")
            plt.xlabel("wavelength (nm)")
            plt.ylabel(f"{t} ($mm^{-1}$)")
            plt.savefig(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}",f"{t}.png"))
            plt.show()
            
        
        plt.plot(used_wl, T2_large[:20], label="large IJV SDS1") # plot large SDS1
        plt.plot(used_wl, T2_small[:20], label="small IJV SDS1") # plot small SDS1
        plt.plot(used_wl, T2_large[20:40], label="large IJV SDS2") # plot large SDS2
        plt.plot(used_wl, T2_small[20:40], label="small IJV SDS2") # plot small SDS2
        plt.title(f"absolute error = {e:.2f}%")
        plt.xlabel("wavelength (nm)")
        plt.ylabel("reflectance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}",f"T2_{idx}.png"))
        plt.show()
        
        plt.plot(used_wl, T1_large[:20], label="large IJV SDS1") # plot large SDS1
        plt.plot(used_wl, T1_small[:20], label="small IJV SDS1") # plot small SDS1
        plt.plot(used_wl, T1_large[20:40], label="large IJV SDS2") # plot large SDS2
        plt.plot(used_wl, T1_small[20:40], label="small IJV SDS2") # plot small SDS2
        plt.title(f"absolute error = {e:.2f}%")
        plt.xlabel("wavelength (nm)")
        plt.ylabel("reflectance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}",f"T1_{idx}.png"))
        plt.show()
        
        plt.plot(used_wl, T2_large[:20], label="T2 large IJV SDS1") # plot large SDS1
        plt.plot(used_wl, T2_small[:20], label="T2 small IJV SDS1") # plot small SDS1
        plt.plot(used_wl, T2_large[20:40], label="T2 large IJV SDS2") # plot large SDS2
        plt.plot(used_wl, T2_small[20:40], label="T2 small IJV SDS2") # plot small SDS2
        plt.plot(used_wl, T1_large[:20], label="T1 large IJV SDS1") # plot large SDS1
        plt.plot(used_wl, T1_small[:20], label="T1 small IJV SDS1") # plot small SDS1
        plt.plot(used_wl, T1_large[20:40], label="T1 large IJV SDS2") # plot large SDS2
        plt.plot(used_wl, T1_small[20:40], label="T1 small IJV SDS2") # plot small SDS2
        plt.title(f"absolute error = {e:.2f}%")
        plt.xlabel("wavelength (nm)")
        plt.ylabel("reflectance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}",f"mergeT1T2_{idx}.png"))
        plt.show()
        
        plt.plot(used_wl, np.log(T2_large[:20]/T2_small[:20]), label="T2 ratio SDS1") # plot T2 ratio SDS1
        plt.plot(used_wl, np.log(T2_large[20:40]/T2_small[20:40]), label="T2 ratio SDS2") # plot T2 ratio SDS2
        plt.plot(used_wl, np.log(T1_large[:20]/T1_small[:20]), label="T1 ratio SDS1") # plot T1 ratio SDS1
        plt.plot(used_wl, np.log(T1_large[20:40]/T1_small[20:40]), label="T1 ratio SDS2") # plot T1 ratio SDS2
        plt.plot(used_wl, np.log(T2_large[:20]/T2_small[:20])-np.log(T1_large[:20]/T1_small[:20]), "r--",label="$\u0394$R SDS1") # delta ratio
        plt.plot(used_wl, np.log(T2_large[20:40]/T2_small[20:40])-np.log(T1_large[20:40]/T1_small[20:40]), "b--", label="$\u0394$R SDS2")
        plt.title(f"Reflectance ratio absolute error = {e:.2f}%")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05,1.0))
        plt.xlabel("wavelength (nm)")
        plt.ylabel("Reflectance Ratio")
        plt.tight_layout()
        plt.savefig(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}",f"Ratio_{idx}.png"),bbox_inches="tight")
        plt.show()
        
        np.save(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}", "parameters.npy"), parameters)
        json_save = {"skin_mus" : list(skin_mus),
                     "fat_mus" : list(fat_mus),
                     "muscle_mus" : list(muscle_mus),
                     "IJV_mus" : list(IJV_mus),
                     "CCA_mus" : list(CCA_mus),
                     "skin_mua" : list(skin_mua),
                     "fat_mua" : list(fat_mua),
                     "muscle_mua" : list(muscle_mua),
                     "IJV_mua" : list(IJV_mua),
                     "CCA_mua" : list(CCA_mua),
                     "IJV_70_mua" : list(IJV_70_mua),
                     "T2_large" : list(T2_large),
                     "T2_small" : list(T2_small),
                     "T1_large" : list(T1_large),
                     "T1_small" : list(T1_small)}
        with open(os.path.join(os.path.join("pic",f"{local_time}",f"{local_time}_{out_acc}",f"{e:.2f}_{idx}", f"truth_{ground_truth:.2f}_parameters.json")), "w") as f:
                  json.dump(json_save, f, indent=4)
#%%   
if __name__ == "__main__":
    seconds = time.time()
    local_time = time.ctime(seconds)
    # train_loader = torch.load("train_loader.pth")
    test_loader = torch.load("test_loader.pth")
    # train model
    model = ANN().cuda()
    
    # input_names = ['delta_log_RMAX_RMIN']
    # output_names = ['delta SO2']
    # input_tensor = torch.tensor(np.random.rand(20)).to(torch.float32).cuda()
    # torch.onnx.export(model,input_tensor, 'ANN.onnx', input_names=input_names, output_names=output_names)
    
    # plot result
    with open('trlog.pkl', 'rb') as f:
        trlog = pickle.load(f)
    min_loss = min(trlog['test_loss'])
    ep = trlog['test_loss'].index(min_loss)
    model = ANN().cuda()
    # model.load_state_dict(torch.load(f"ep_{ep}_loss_{min_loss}.pth"))
    model.load_state_dict(torch.load("ep_949_loss_0.00016349506972111069.pth"))
    
    with open("mus_bound.json", "r") as f:
        mus_bound = json.load(f)
    with open("mua_bound.json", "r") as f:
        mua_bound = json.load(f)
    pred()
    
    