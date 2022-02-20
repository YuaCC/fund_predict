import torch
from dataset import dataloader
from model.net import Net
from torchvision.transforms import ToTensor
from torch.optim import Adam,SGD
import numpy as np

if __name__=='__main__':

    args = [['files2.txt','best_model1.pkl'],['files1.txt','best_model2.pkl']]

    results = []
    for filea,modelname in args:
        model = Net()
        model.load_state_dict(torch.load(modelname))
        model=model.cuda()
        val_loader = dataloader('./data/dataset',f'./data/{filea}',"test",batch_size=64)
        model = model.eval()
        for x, y in val_loader:
            x = x.cuda()
            scores = model(x)
            scores_list = scores.detach().cpu().numpy().tolist()
            results.extend(zip(scores_list,y))
    results = sorted(results,key=lambda x:x[0],reverse=True)
    print(results)







