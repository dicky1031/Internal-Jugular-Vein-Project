import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
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

# #%% model2
# class ANN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(201, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#             )
        
#     def forward(self, x):
#         return self.net(x)

#%% train model
def train():
    trlog = {}
    trlog['train_loss'] = []
    trlog['test_loss'] = []
    min_loss = 100000
    for ep in range(epoch):
        model.train()
        tr_loss = 0
        for batch_idx, (data,target,parameters) in enumerate(train_loader):
            data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
            optimizer.zero_grad()
            output = model(data)
            target = target.view(-1,1)
            # loss = torch.sqrt(torch.square((output-target)/target).mean())
            loss = criterion(output,target)
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
            
 
            if batch_idx % int(0.1*len(train_loader)) == 0:
                print(f"[train] ep:{ep}/{epoch}({100*ep/epoch:.2f}%) batch:{batch_idx}/{len(train_loader)}({100*batch_idx/len(train_loader):.2f}%)\
                      loss={tr_loss/(batch_idx+1)}")
        # scheduler.step(tr_loss/(batch_idx+1))
        trlog['train_loss'].append(tr_loss/len(train_loader))
        min_loss = test(trlog,ep,min_loss)
        
    
    return trlog

def test(trlog,ep,min_loss):
    model.eval()
    ts_loss = 0
    for batch_idx, (data,target,parameters) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        # optimizer.zero_grad()
        output = model(data)
        target = target.view(-1,1)
        # loss = torch.sqrt(torch.square((output-target)/target).mean())
        loss = criterion(output,target)
        ts_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        
        
    print(f"[test] batch:{batch_idx}/{len(test_loader)}({100*batch_idx/len(test_loader):.2f}%) loss={ts_loss/len(test_loader)}")
    trlog['test_loss'].append(ts_loss/len(test_loader))
    
    if min_loss > ts_loss/len(test_loader):
        min_loss = ts_loss/len(test_loader)
        torch.save(model.state_dict(), f"ep_{ep}_loss_{min_loss}.pth")
        with open('trlog.pkl', 'wb') as f:
            pickle.dump(trlog, f)
            
    return min_loss

#%% plot pred
def pred():
    SO2 = [(i-70)/100 for i in range(40,91,1)]
    SO2_text = [f'{(i-70)}%' for i in range(40,91,1)]
    model.eval()
    error = []
    max_error = 0
    for batch_idx, (data,target, parameters) in enumerate(test_loader):
        data,target = data.to(torch.float32).cuda(), target.to(torch.float32).cuda()
        # optimizer.zero_grad()
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
        plt.plot(target,output, 'r.', markersize=5)
        plt.plot(target,target,'b')
    error = np.array(error)
    std = np.std(error)
    plt.title(f"based on SO2=70% \nmean error:{np.mean(error):.2f}% std:{std:.2f}% \nmax error:{max(error):.2f}%")
    # plt.xticks(SO2, SO2_text)
    # plt.yticks(SO2, SO2_text)
    plt.xlabel("truth $\u0394$SO2")
    plt.ylabel("predict $\u0394$SO2")
    plt.savefig("RMSPE.png")
    plt.show()

#%%   
if __name__ == "__main__":
    root = "prediction_ANN_input.npy"
    batch_size = 256
    # split data setting
    # set testing data size
    test_split = 0.1
    # need shuffle or not
    shuffle_dataset = True
    # random seed of shuffle 
    random_seed = 703
    # dataset = dataload(root)
    train_dataset = dataload(root="prediction_ANN_input.npy")
    test_dataset = dataload(root="test_prediction_ANN_input.npy")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    small_train_loader, smallest_train_loader = data_preprocess(train_dataset, batch_size, test_split, shuffle_dataset, random_seed)
    # torch.save(train_loader, "train_loader.pth")
    torch.save(smallest_train_loader, "smallest_train_loader.pth")
    torch.save(test_loader, "test_loader.pth")
    
    # train model
    model = ANN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = nn.MSELoss()
    epoch = 2000
    trlog = train()
    with open('trlog.pkl', 'wb') as f:
        pickle.dump(trlog, f)
    
    # plot result
    with open('trlog.pkl', 'rb') as f:
        trlog = pickle.load(f)
    min_loss = min(trlog['test_loss'])
    ep = trlog['test_loss'].index(min_loss)
    model = ANN().cuda()
    model.load_state_dict(torch.load(f"ep_{ep}_loss_{min_loss}.pth"))
    # model.load_state_dict(torch.load("ep_54_loss_0.00076047529881971.pth"))
    pred()