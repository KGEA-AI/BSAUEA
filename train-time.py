import argparse
import itertools

import torch,gc
import torch.nn as nn
import torch.nn.functional as F
import os
from model import ELTEA
from data import Data
from loss import Loss
from utils import setup_seed,  get_hits_COS, get_hits_Sinkhorn
from utils2 import *
import random
import pickle
    
def train(model, criterion, optimizer, data,unspervised=False):
    model.train()

    x1, x2 = model(data.x1, data.x2, data.edge1, data.edge2)
    if unspervised:
        loss = criterion(x1, x2)
        
    else:
        loss = criterion(x1, x2, data.train_set)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    return loss
    

def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1, x2 = model(data.x1, data.x2, data.edge1, data.edge2)
    return x1, x2

def getMemory(text):
    print(text)
    gpu_stats = torch.cuda.memory_stats()
    gpu_memory_used = gpu_stats["allocated_bytes.all.current"] / 1024**3
    gpu_memory_free = gpu_stats["reserved_bytes.all.current"] / 1024**3
    gpu_memory_total = gpu_stats["reserved_bytes.all.peak"] / 1024**3

    print(f"GPU Memory Used: {gpu_memory_used:.2f} GB")
    print(f"GPU Memory Free: {gpu_memory_free:.2f} GB")
    print(f"GPU Memory Total: {gpu_memory_total:.2f} GB")

def test(x1, x2, test_set,test_time, name):

    with torch.no_grad():
        Cos = get_hits_COS(x1, x2, test_set,test_time)
        Sinkhorn = get_hits_Sinkhorn(x1, x2, test_set,test_time)
        print(f'{name} Cos: {Cos}, Sinkhorn: {Sinkhorn}')
   

    
def test_all(x1, x2, data, unspervised=False):
    if not unspervised:
        test(x1, x2, data.train_set,data.train_time, 'Train')
    test(x1, x2, data.eval_set,data.eval_time, 'Eval')
    test(x1, x2, data.test_set,data.test_time, 'Test')
    
data = Data(args.data, args.embs, args.rate, args.seed, args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--data", default="data/YAGO-WIKI50K")
#    parser.add_argument("--data", default="data/ICEWS05-15")
    parser.add_argument("--embs", default="data/glove.6B.300d.txt")
    parser.add_argument("--rate", type=float, default=1)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--test_epoch", type=int, default=10)
    parser.add_argument("--unspervised", default=False)
    args = parser.parse_args()

    # 原有代码
    setup_seed(args.seed)
    data = Data(args.data, args.embs, args.rate, args.seed, args.device)

    #  插入此段（结构相似判断+BERT编码）
    from bert_encoder import BERTNameEncoder
    from structure_gate import should_enable_gcn

    # 加载实体名称（你也可以调用 data.path 来拼接完整路径）
    name_path_1 = os.path.join(args.data, 'ent_names_1')
    name_path_2 = os.path.join(args.data, 'ent_names_2')
    names_1 = [line.strip().split('\t')[1] if '\t' in line else '' for line in open(name_path_1, encoding='utf-8')]
    names_2 = [line.strip().split('\t')[1] if '\t' in line else '' for line in open(name_path_2, encoding='utf-8')]

    bert_encoder = BERTNameEncoder().to(args.device)
    with torch.no_grad():
        x1_bert = bert_encoder(names_1).to(args.device)
        x2_bert = bert_encoder(names_2).to(args.device)

    # 判断结构相似度是否超过阈值，决定是否使用GCN
    struct_sim = 0.81  # TODO: 动态读取或配置
    if should_enable_gcn(struct_sim):
        print(" GCN is enabled.")
        data.x1 = model.gcn1(x1_bert, data.edge1)
        data.x2 = model.gcn2(x2_bert, data.edge2)
    else:
        print(" GCN is skipped.")
        data.x1 = x1_bert
        data.x2 = x2_bert

    #  时间矩阵生成
#     train_pair,dev_pair,all_pair,adj_matrix,adj_features,rel_features,time_dict = load_data(args.data+"/")
#     print('读取完毕')
#     rest_set_1 = [e1 for e1, e2 in dev_pair]
#     rest_set_2 = [e2 for e1, e2 in dev_pair]
#     np.random.shuffle(rest_set_1) 
#     np.random.shuffle(rest_set_2)
#     t1 = [list2dict(time_dict[e1]) for e1 in rest_set_1]
#     t2 = [list2dict(time_dict[e2]) for e2 in rest_set_2]
#     print('计算相似度矩阵:')
#     m = thread_sim_matrix(t1,t2)
#     with open(m_filename,"wb") as f:
#         pickle.dump(m,f)
#         print("储存相似度矩阵成功:",m_filename)

    

    with open("m_ICEWS05-15.pkl","rb") as f:
        m=pickle.load(f)
        print("读取相似度矩阵成功:",m.shape)
    m=torch.from_numpy(m)
    
    data.train_time = m[:int(args.rate*each_num),:int(args.rate*each_num)]
    data.eval_time = m[int(args.rate*each_num):int((args.rate+0.05)*each_num),int(args.rate*each_num):int((args.rate+0.05)*each_num)]
    data.test_time = m[int(args.rate*each_num):,int(args.rate*each_num):]
    
    print(data.x1.shape,data.x2.shape,data.edge1.shape,data.edge2.shape)
    
  
    if args.unspervised:
        print("无监督学习",data.train_set.shape,data.test_set.shape)
        data.test_set = torch.cat([data.train_set, data.test_set])
        #加时间调整
#         print(data.test_set.shape)

#         data.test_time = m
#         print(data.test_time.shape)
        
    data.x1.requires_grad_(), data.x2.requires_grad_()
    #test_all(data.x1, data.x2, data, args.unspervised)
    size_e1, size_e2 = data.x1.size(0), data.x2.size(0)
    size_r1, size_r2 = max(data.edge1[1])+1, max(data.edge2[1])+1
    model = ELTEA(size_e1, size_r1, size_e2, size_r2).to(args.device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x1, data.x2])))
    criterion = Loss(args.k)
    maxx = 0
    
    best_h1, best_x1, best_x2 = 0, None, None
    for epoch in range(args.epoch):
        loss = train(model, criterion, optimizer, data,args.unspervised)
        print(f'----------Epoch: {epoch+1}/{args.epoch}, Loss: {loss:.4f}----------\r', end='')
        x1, x2 = get_emb(model, data)
       
        h1 = get_hits_COS(x1, x2, data.eval_set,data.eval_time)[0]
        if h1 > best_h1:
            best_h1 = h1
            best_x1, best_x2 = x1.cpu(), x2.cpu()

        if (epoch+1)%args.test_epoch == 0:
            print()
#             test(x1, x2, data.train_set,data.train_time, 'Train') 无监督学习请注释train
            test(x1, x2, data.eval_set,data.eval_time, 'Eval')
#             test_all(x1, x2, data, args.unspervised)  每10轮测试一次 为节约性能选择性注释

    print('----------Final Results----------')
    
    x1, x2 = best_x1.to(args.device), best_x2.to(args.device)
    test_all(x1, x2, data, args.unspervised)