'''
Functions for DA algorithm
'''
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import time


def batched_rowwise_topk_slow(a: torch.Tensor, k_vec: torch.Tensor):
    n, m = a.shape
    sorted_values, sorted_indices = torch.sort(a, dim=1, descending=True)

    col_idx = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    k_mask = col_idx < k_vec.unsqueeze(1)  # (n, m)
    topk_values = sorted_values.masked_fill(~k_mask, float('-inf'))
    topk_indices = sorted_indices.masked_fill(~k_mask, -1)
    topk_indices[topk_values == float('-inf')] = -1
    return topk_values, topk_indices


def batched_rowwise_topk(a: torch.Tensor, k_vec: torch.Tensor):
    n, m = a.shape
    kmax = k_vec.max().item()
    topk_values, topk_indices = torch.topk(a, kmax, dim=1, largest=True, sorted=False)
    col_idx = torch.arange(kmax, device=a.device).unsqueeze(0)
    k_mask = col_idx < (torch.ones(len(k_vec), dtype=torch.int64) * k_vec).unsqueeze(1)
    topk_values = topk_values.masked_fill(~k_mask, float('-inf'))
    topk_indices = topk_indices.masked_fill(~k_mask, -1)
    topk_indices[topk_values == float('-inf')] = -1

    return topk_values, topk_indices


def DA_algorithm(Value1, Value2, quota1=None, quota2=None, propose=1, break_tie_value=1e-5):
    Value1 = Value1.float().cpu().clone()
    Value2 = Value2.float().cpu().clone()
    if propose == 2:
        Value1, Value2 = Value2, Value1
        quota1, quota2 = quota2, quota1
    assert (Value1.shape[1]) == (Value2.shape[0]), "Column number of Value1 must be the same as Row number of Value2"
    assert (Value1.shape[0]) == (Value2.shape[1]), "Row number of Value1 must be the same as Column number of Value2"
    if quota1 is None:
        quota1 = torch.ones(Value1.shape[0], dtype=torch.int64)
    else:
        quota1 = torch.tensor(quota1, dtype=torch.int64)
    if quota2 is None:
        quota2 = torch.ones(Value2.shape[0], dtype=torch.int64)
    else:
        quota2 = torch.tensor(quota2, dtype=torch.int64)
    quota1.reshape(Value1.shape[0])
    quota2.reshape(Value2.shape[0])
    assert (Value1.shape[0]) == (len(quota1)), "Row number of Value1 must be the same as length of quota1"
    assert (Value2.shape[0]) == (len(quota2)), "Row number of Value2 must be the same as length of quota2"
    if (Value1.shape[0]==0) or (Value1.shape[1]==0):
        return torch.empty(Value1.shape[0],Value1.shape[1],dtype=bool)

    while (True):
        Value1_propose = batched_rowwise_topk(Value1, quota1)[1]
        accept_matrix = torch.zeros((Value1.shape[0], Value1.shape[1]), dtype=bool)
        for i in range(Value2.shape[0]):
            propose_to_i = torch.nonzero(torch.any(torch.eq(Value1_propose, i), dim=1)).squeeze(1)
            if len(propose_to_i) > 0:
                accept_by_i = propose_to_i[torch.topk(Value2[i, propose_to_i], min(len(propose_to_i), quota2[i]))[1]]
                accept_matrix[accept_by_i, i] = True
                reject_by_i = list(set(propose_to_i.numpy()) - set(accept_by_i.numpy()))
                Value1[reject_by_i, i] = -float('inf')
        # break or not?
        matched_number = accept_matrix.sum(dim=1)
        Value1_full_quota = set(torch.where(matched_number == quota1)[0].numpy())
        temp_matrix = (Value1 == -float('inf'))
        temp_satisfied = torch.nonzero(torch.all((temp_matrix | accept_matrix), dim=1)).squeeze(1)
        temp_satisfied = set(temp_satisfied.numpy())
        if len(Value1_full_quota.union(temp_satisfied)) == (Value1.shape[0]):
            break
    if propose == 2:
        return accept_matrix.t()
    else:
        return accept_matrix