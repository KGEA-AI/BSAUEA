import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_hits_COS(x1, x2, pair,time, Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    alpha=0.1
    if pair_num>5000:
        S = torch.mm(x1[pair[:, 0]], x2[pair[:, 1]].t()).cpu()
    else:
        S = torch.mm(x1[pair[:, 0]], x2[pair[:, 1]].t())
        time=time.to(S.device)

#     S=torch.add((1-alpha)*S,alpha*time)
    
    Hks = []
    for k in Hn_nums:
        pred_topk= S.topk(k)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        Hks.append(round(Hk*100, 2))
    rank = torch.where(S.sort(descending=True)[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = round((1/(rank+1)).mean().item(), 3)
    return Hks+[MRR]


def get_hits_Sinkhorn(x1, x2, pair,time, Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    alpha=0.1
    if pair_num>5000:
        S = torch.mm(x1[pair[:, 0]], x2[pair[:, 1]].t()).cpu()
    else:
        S = torch.mm(x1[pair[:, 0]], x2[pair[:, 1]].t())
        time=time.to(S.device)
    #print(time.shape)
#     S=torch.add((1-alpha)*S,alpha*time)
    S = torch.exp(S*50)
    
    for i in range(10):
        S /= torch.sum(S, dim=0, keepdims=True)
        S /= torch.sum(S, dim=1, keepdims=True)
    Hks = []
    for k in Hn_nums:
        pred_topk= S.topk(k)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        Hks.append(round(Hk*100, 2))
    rank = torch.where(S.sort(descending=True)[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = round((1/(rank+1)).mean().item(), 3)
    return Hks+[MRR]
