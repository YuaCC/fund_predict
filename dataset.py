import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose


class FundDataset(Dataset):
    def __init__(self, data_folder, data_file, transform, mode, use_cache=True):
        super(FundDataset, self).__init__()
        self.data_folder = data_folder
        with open(data_file, "r") as f:
            self.files = [line.strip() for line in f.readlines()]
        self.transform = transform
        self.mode = mode
        self.use_cache = use_cache
        if self.use_cache:
            print('loading dataset...')
            self.cache = []
            for idx, f in enumerate(self.files):
                f = os.path.join(self.data_folder, f)
                data = np.loadtxt(f, dtype=str, delimiter=",")
                data = data[:, 2].astype(np.float)
                data = data[::-1].copy()
                self.cache.append(data)
                if idx % 1000 == 0:
                    print(f'{idx}/{len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.use_cache:
            data = self.cache[idx]
        else:
            f = os.path.join(self.data_folder, self.files[idx])
            data = np.loadtxt(f, dtype=str, delimiter=",")
            data = data[:, 2].astype(np.float)
            data = data[::-1].copy()
        if self.mode == "train":
            if len(data) < 162:
                raise ValueError(f'length < 162 {self.files[idx]}', )
            if self.transform:
                data = self.transform(data)
            data = data.unsqueeze(0)
            x, y = data[:, 81:], data[:, :81]
            y0 = y[:, -1]
            y = torch.mean(y, dim=1)
            return x.float(), y.float(), y0.float()
        elif self.mode == "val":
            if len(data) < 162:
                raise ValueError(f'length < 162 {self.files[idx]}', )
            if self.transform:
                data = self.transform(data)
            data = data.unsqueeze(0)
            x, y = data[:, 81:], data[:, :81]
            y0 = y[:, -1]
            y = torch.mean(y, dim=1)
            return x.float(), y.float(), y0.float()
        elif self.mode == "test":
            if len(data) < 81:
                raise ValueError(f'length < 81 {self.files[idx]}', )
            if self.transform:
                data = self.transform(data)
            x = data.unsqueeze(0)
            return x.float(), self.files[idx]
        else:
            raise ValueError(f"mode {self.mode} not supported yet")


class TrainSpliter:
    def __init__(self, length_min):
        self.length_min = length_min

    def __call__(self, data):
        if len(data) < self.length_min:
            raise ValueError(f'length < {self.length_min} ', )
        length = np.random.randint(self.length_min, len(data) + 1)
        beg = np.random.randint(len(data) - length + 1)
        data = data[beg:beg + length]
        return data


class ValSpliter:
    def __init__(self, length_min):
        self.length_min = length_min

    def __call__(self, data):
        if len(data) < self.length_min:
            raise ValueError(f'length < {self.length_min} ', )
        beg = np.random.randint(len(data) - self.length_min + 1)
        data = data[beg:]
        return data


class Scaler:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        r = (np.random.rand() * 2 - 1) * self.a
        r = np.power(2, r)
        return x * r


def trainval_collate_fn(data):
    length = max(d[0].shape[1] for d in data)
    length = max(length, 727)
    batch = len(data)
    x = torch.zeros((batch, 1, length))
    for i in range(batch):
        l = data[i][0].shape[1]
        x[i][:, :l] = data[i][0]
    y = torch.stack([d[1] for d in data], 0)
    y0 = torch.stack([d[2] for d in data], 0)
    return x, y, y0


def test_collate_fn(data):
    length = max(d[0].shape[1] for d in data)
    length = max(length, 727)
    batch = len(data)
    x = torch.zeros((batch, 1, length))
    for i in range(batch):
        l = data[i][0].shape[1]
        x[i][:, :l] = data[i][0]
    y = [d[1] for d in data]
    return x, y


def dataloader(data_folder, data_file, mode, batch_size=16):
    if mode == "train":
        transform = Compose([
            Scaler(1),
            TrainSpliter(243),
            torch.tensor,
        ])
        dataset = FundDataset(data_folder, data_file, transform, mode)
        return DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
                          collate_fn=trainval_collate_fn)
    elif mode == "val":
        transform = Compose([
            ValSpliter(243),
            torch.tensor,
        ])
        dataset = FundDataset(data_folder, data_file, transform, mode)
        return DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True,
                          collate_fn=trainval_collate_fn)
    elif mode == "test":
        transform = Compose([
            torch.tensor,
        ])
        dataset = FundDataset(data_folder, data_file, transform, mode)
        return DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=test_collate_fn)
    else:
        raise ValueError(f"mode {mode} not supported yet")


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
