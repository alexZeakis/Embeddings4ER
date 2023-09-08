#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from time import time
import json
from utils import vectorizers, cases
import sys
import os



def topk(x, k):
    out = []
    for no, xx in enumerate(x):
        out.append((no, xx))
    out = sorted(out, key=lambda x: x[1])
    return out[:k]

def find_exact_nns(tensor1, tensor2, k, gpu=False):
    tensor11 = torch.Tensor(tensor1).cuda()
    tensor22 = torch.Tensor(tensor2).cuda()
    
    dists = torch.cdist(tensor11, tensor22, p=2)

    topk_dists = torch.topk(dists, k, largest=False)
    results = []
    for no, res_list in enumerate(topk_dists.indices):
        for res in res_list:
            results.append((no, res.item()))
    results = set(results)
    return results
    


def calc_recall(true, preds):
    return len(true & preds) / len(true)

def calc_precision(true, preds):
    return len(true & preds) / len(preds)


input_dir = sys.argv[1]
emb_dir = sys.argv[2]
log_file = sys.argv[3] + 'complementarity.txt'

os.makedirs(os.path.dirname(log_file), exist_ok=True)

ks = [1, 10]
gpu = True

scores2 = []

f = open(log_file, 'w')

for nocase, (data1, data2, ground_file, sep, dir, cols) in enumerate(cases):
    nocase = f'D{nocase+1}'
    print()
    print(nocase)
    #print(noc, data1, data2, ground_file, sep, dir, cols)
    
    ground_file = '{}{}/{}.csv'.format(input_dir, dir, ground_file)
    ground_df = pd.read_csv(ground_file, sep=sep)
    ground_results = set(ground_df.apply(lambda x: (x[0], x[1]), axis=1).values)
    
    for nocol, (col1, col2) in enumerate(cols):
        if nocol != 2:
            continue
        for vec in vectorizers:
            file1 = '{}{}/{}_{}_{}.csv'.format(emb_dir, dir, data1, col1, vec)
            file2 = '{}{}/{}_{}_{}.csv'.format(emb_dir, dir, data2, col2, vec)
            
            df1 = pd.read_csv(file1, header=None, index_col=0).values
            df2 = pd.read_csv(file2, header=None, index_col=0).values
            if gpu:
                df1 = torch.Tensor(df1)
                df2 = torch.Tensor(df2)
            
            for k in ks:
                print('\t{} {} {}\r'.format(nocol, vec, k), end='')
                
                                
                #exact - input2query
                t1 = time()
                results = find_exact_nns(df2, df1, k, gpu)
                t2 = time()
                results = [(y,x) for x,y in results] #reverse the input to query results to become (q_id, in_id)
                
                log = {'vec': vec, 'case':nocase, 'ranks': results, 'k': k}
                f.write(json.dumps(log)+'\n')
                
                #scores2.append((nocase, nocol, vec, k, rec_qi, rec_iq))
            #break
        #break
    # break
    

