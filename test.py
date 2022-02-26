import torch

from dataset import dataloader
from model.net import Net

if __name__ == '__main__':
    files = './data/files.txt'
    dataset_folder = './data/dataset'
    weight_file = 'best_model.pkl'
    results = []
    results_map = {}
    my_funds = ['519674','001933','001822','006401','004233',
                '011322','001306','002256','006080','001606',
                '519158','162204','110013','004391','519001',
                '000595','519035','240008','040005','260104',
                '110011']

    model = Net()
    model.load_state_dict(torch.load(weight_file))
    model = model.cuda()
    rain_loader,val_loader,test_loader = dataloader(dataset_folder, files, 64)
    model = model.eval()
    for x, y in test_loader:
        x = x.cuda()
        scores = model(x)
        scores_list = scores[:,0].detach().cpu().numpy().tolist()
        results.extend(zip(scores_list, y))
    results = sorted(results, key=lambda x: x[0], reverse=True)

    for score,filename in results:
        print(f'{score},{filename}')
        results_map[filename] = score
    print("my funds")
    for fund_id in my_funds:
        print(f"{fund_id},{results_map[f'{fund_id}.csv']}")

