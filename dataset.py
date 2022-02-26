import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose


class FundDataset(Dataset):
    def __init__(self, data, files , transform, mode):
        super(FundDataset, self).__init__()
        self.data = data
        self.files = files
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.mode == "train" or self.mode=='val':
            if len(data) < 243:
                raise ValueError(f'length < 243 {self.files[idx]}', )
            data = self.transform(data)
            x = data.unsqueeze(0)
            return x
        elif self.mode == "test":
            if len(data) < 81:
                raise ValueError(f'length < 81 {self.files[idx]}', )
            data = self.transform(data)
            x = data.unsqueeze(0)
            return x.float(), self.files[idx]
        else:
            raise ValueError(f"mode {self.mode} not supported yet")


class Scaler:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        r = (np.random.rand() * 2 - 1) * self.a
        r = np.power(2, r)
        return x * r


def train_collate_fn(data):
    max_length = max(d.shape[1] for d in data)
    min_length = min(d.shape[1] for d in data)
    batch = len(data)
    x = torch.zeros((batch, 1, max_length))
    for i in range(batch):
        l = data[i].shape[1]
        x[i][:, :l] = data[i]
    beg = np.random.randint(81,min_length-162+1)
    x = x[:,:,beg:]
    x,y = x[:,:,81:],x[:,:,:81]
    y0 = y[:,:,80]
    y = torch.mean(y,dim=2)
    return x, y, y0


def val_collate_fn(data):
    max_length = max(d.shape[1] for d in data)
    min_length = min(d.shape[1] for d in data)
    batch = len(data)
    x = torch.zeros((batch, 1, max_length))
    for i in range(batch):
        l = data[i].shape[1]
        x[i][:, :l] = data[i]
    x,y = x[:,:,81:],x[:,:,:81]
    y0 = y[:,:,80]
    y = torch.mean(y,dim=2)
    return x, y, y0


def test_collate_fn(data):
    max_length = max(d[0].shape[1] for d in data)
    min_length = min(d[0].shape[1] for d in data)
    batch = len(data)
    x = torch.zeros((batch, 1, max_length))
    for i in range(batch):
        l = data[i][0].shape[1]
        x[i][:, :l] = data[i][0]
    y = [d[1] for d in data]
    return x, y


def dataloader(data_folder, data_file, batch_size=16):
    with open(data_file, "r") as f:
        files = [line.strip() for line in f.readlines()]
    datas = []
    print('loading files')
    for idx, f in enumerate(files):
        f = os.path.join(data_folder, f)
        data = np.loadtxt(f, dtype=str, delimiter=",")
        data = data[:, 2].astype(np.float)
        data = data[::-1].copy()
        datas.append(data)
        if idx % 1000 == 0:
            print(f'{idx}/{len(files)}')
    train_transform = Compose([
        Scaler(1),
        torch.tensor,
    ])
    dataset = FundDataset(datas, files, train_transform, 'train')
    train_loader =  DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
                      collate_fn=train_collate_fn)
    val_transform = Compose([
        torch.tensor,
    ])
    dataset = FundDataset(datas, files, val_transform, 'val')
    val_loader= DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True,
                      collate_fn=val_collate_fn)
    test_transform = Compose([
        torch.tensor,
    ])
    dataset = FundDataset(datas, files, test_transform, 'test')
    test_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True,
                     collate_fn=test_collate_fn)
    return train_loader,val_loader,test_loader


def cal_mean_std():
    import numpy as np
    transform = Scaler(1)
    dataset1 = FundDataset('./data/dataset', './data/files1.txt', transform, "val")
    dataset2 = FundDataset('./data/dataset', './data/files2.txt', transform, "val")
    Ex = 0
    Ex2 = 0
    for d in dataset1:
        x = d[0].numpy()
        Ex += x.mean()
        Ex2 += (x * x).mean()
    for d in dataset2:
        x = d[0].numpy()
        Ex += x.mean()
        Ex2 += (x * x).mean()
    Ex /= len(dataset1) + len(dataset2)
    Ex2 /= len(dataset2) + len(dataset2)
    std = np.sqrt(Ex2 - Ex * Ex)
    print(Ex)
    print(Ex2)
    print(std)
    return Ex, std


if __name__ == '__main__':
    cal_mean_std()
