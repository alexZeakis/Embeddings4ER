#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import time
import numpy as np
import sys
from utils import vectorizers, cases

def find_exact_nns(tensor1, tensor2, k):
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


data_dir = sys.argv[1]
emb_dir = sys.argv[2]
log_file = sys.argv[3] + 'blocking.csv'
ks = [10, 5, 1]

scores2 = []

os.makedirs(os.path.dirname(log_file), exist_ok=True)
for nocase, (data1, data2, ground_file, sep, dir, cols) in enumerate(cases):
    nocase = f'D{nocase+1}'
    print()
    print(nocase)
    #print(noc, data1, data2, ground_file, sep, dir, cols)
    
    ground_file = '{}{}/{}.csv'.format(data_dir, dir, ground_file)
    print(ground_file)
    ground_df = pd.read_csv(ground_file, sep=sep)
    ground_results = set(ground_df.apply(lambda x: (x[0], x[1]), axis=1).values)
    
    for nocol, (col1, col2) in enumerate(cols):
        if nocol == 2:
            continue
        for vec in vectorizers:
            file1 = '{}{}/{}_{}_{}.csv'.format(emb_dir, dir, data1, col1, vec)
            file2 = '{}{}/{}_{}_{}.csv'.format(emb_dir, dir, data2, col2, vec)
            
            df1 = pd.read_csv(file1, header=None, index_col=0).values
            df2 = pd.read_csv(file2, header=None, index_col=0).values
            df1 = torch.Tensor(df1)
            df2 = torch.Tensor(df2)
            
            for k in ks:
                print('\t{} {} {}\r'.format(nocol, vec, k), end='')
                                                
                #exact - input2query
                t1 = time()
                results = find_exact_nns(df2, df1, k)
                t2 = time()
                results = set([(y,x) for x,y in results]) #reverse the input to query results to become (q_id, in_id)
                recall = calc_recall(ground_results, results)
                precision = calc_precision(ground_results, results)
                scores2.append((nocase, nocol, vec, k, 'i2q', 'exact', recall, precision, t2-t1))
                
                #scores2.append((nocase, nocol, vec, k, rec_qi, rec_iq))
            #break
        #break
    #break
    
results = pd.DataFrame(scores2, columns=['Case', 'Columns', 'Vectorizer', 'k', 'Direction',
                                         'Exact', 'Recall', 'Precision', 'Time'])    
results.to_csv(log_file, header=True, index=True)
results

