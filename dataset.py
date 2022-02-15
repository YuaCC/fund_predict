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
        # self.mean = 1.541989
        # self.std = 1.085787
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
            offset = np.random.randint(0,len(data)-162+1)
            data = data[offset:offset+162]
            x,y = data[:81],data[81:]
            x = torch.tensor(x)
            x=x.unsqueeze(0)
            if self.transform:
                x=self.transform(x)
            y0 = torch.tensor([y[0],])
            y = torch.tensor([np.mean(y[:9]),np.mean(y[:27]),np.mean(y[:81])])
            return x.float(),y.float(),y0.float()
        else:
            if len(data)<81:
                raise ValueError(f'length < 81 {self.files[idx]}',)
            else:
                data= data[-81:]
            data = torch.tensor(data)
            data = data.unsqueeze(0)
            if self.transform:
                data=self.transform(data)
            return data.float(),self.files[idx]


def val_collate_fn(data):
    x=torch.stack([d[0] for d in data],0)
    y=[d[1] for d in data]
    return x,y


def dataloader(data_folder,data_file,transform,training,batch_size=16):
    dataset = FundDataset(data_folder,data_file,transform,training)
    if training:
        return DataLoader(dataset,batch_size,shuffle=True)
    else:
        return DataLoader(dataset,batch_size,shuffle=True,collate_fn=val_collate_fn)


if __name__=='__main__':
    from torchvision.transforms import Compose,ToTensor,Normalize
    transform = Compose([])
    dataset = dataloader('./data/dataset','./data/files1.txt',transform,False)
    for d in dataset:
        print(d)
        break


