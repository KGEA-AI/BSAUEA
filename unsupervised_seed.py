import torch
import numpy as np
import torch.nn.functional as F

def compute_similarity(glove_sim, time_sim, alpha=0.5):
    return alpha * glove_sim + (1 - alpha) * time_sim

def extract_confident_pairs(sim_matrix, threshold=0.5):
    top1 = torch.argmax(sim_matrix, dim=1)
    second = sim_matrix.topk(2, dim=1).values[:, 1]
    confidence = sim_matrix[torch.arange(len(top1)), top1] - second
    mask = confidence > threshold
    pairs = [(i, top1[i].item()) for i in range(len(top1)) if mask[i]]
    return torch.tensor(pairs)

def sinkhorn_normalization(S, T=0.05, q=20):
    for _ in range(q):
        S = S / S.sum(dim=1, keepdim=True)
        S = S / S.sum(dim=0, keepdim=True)
    return S
