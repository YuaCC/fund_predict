from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import torch


class FundDataset(Dataset):
    def __init__(self,data_folder,data_file,transform,training):
        super(FundDataset, self).__init__()
        self.data_folder = data_folder
        with open(data_file,"r") as f:
            self.files = [line.strip() for line in f.readlines()]
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = os.path.join(self.data_folder,self.files[idx])
        data = np.loadtxt(f,dtype=str,delimiter=",")
        data = data[:,2].astype(np.float)
        # data = (data - self.mean)/self.std
        if self.training:
            if len(data)<162:
                raise ValueError(f'length < 162 {self.files[idx]}',)
            length = np.random.randint(162,len(data)+1)
            beg = np.random.randint(len(data)-length+1)
            data = data[beg:beg+length]
            data = data[::-1].copy()

            x,y = data[81:],data[:81]
            x = torch.tensor(x)
            x = x.unsqueeze(0)
            if self.transform:
                x=self.transform(x)
            y0 = torch.tensor([y[-1],])
            y = torch.tensor([np.mean(y),])
            return x.float(),y.float(),y0.float()
        else:
            if len(data)<81:
                raise ValueError(f'length < 81 {self.files[idx]}',)
            data = data[::-1].copy()
            data = torch.tensor(data)
            data = data.unsqueeze(0)
            if self.transform:
                data=self.transform(data)
            return data.float(),self.files[idx]


def train_collate_fn(data):
    length = max(d[0].shape[1] for d in data)
    length = max(length,727)
    batch = len(data)
    x = torch.zeros((batch,1,length))
    for i in range(batch):
        l = data[i][0].shape[1]
        x[i][:,:l] = data[i][0]
    y=torch.stack([d[1] for d in data],0)
    y0=torch.stack([d[2] for d in data],0)
    return x,y,y0


def val_collate_fn(data):
    length = max(d[0].shape[1] for d in data)
    length = max(length,727)
    batch = len(data)
    x = torch.zeros((batch,1,length))
    for i in range(batch):
        l = data[i][0].shape[1]
        x[i][:,:l] = data[i][0]
    y=[d[1] for d in data]
    return x,y


def dataloader(data_folder,data_file,transform,training,batch_size=16):
    dataset = FundDataset(data_folder,data_file,transform,training)
    if training:
        return DataLoader(dataset,batch_size,shuffle=True,collate_fn=train_collate_fn)
    else:
        return DataLoader(dataset,batch_size,shuffle=True,collate_fn=val_collate_fn)

def cal_mean_std():
    from torchvision.transforms import Compose,ToTensor,Normalize
    import numpy as np
    transform = Compose([])
    dataset1 = FundDataset('./data/dataset','./data/files1.txt',transform,False)
    dataset2 = FundDataset('./data/dataset','./data/files2.txt',transform,False)
    Ex = 0
    Ex2 = 0
    for d in dataset1:
        x=d[0].numpy()
        Ex += x.mean()
        Ex2 += (x*x).mean()
    for d in dataset2:
        x=d[0].numpy()
        Ex += x.mean()
        Ex2 += (x*x).mean()
    Ex /= len(dataset1)+len(dataset2)
    Ex2 /= len(dataset2)+len(dataset2)
    std = np.sqrt(Ex2-Ex*Ex)
    print(Ex)
    print(Ex2)
    print(std)
    return Ex,std

if __name__=='__main__':
    cal_mean_std()




