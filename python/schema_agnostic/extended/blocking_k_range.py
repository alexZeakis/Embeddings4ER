#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import statistics
import json
from utils import vectorizers, cases
import sys
import os

input_dir = sys.argv[1]
emb_dir = sys.argv[2]
log_file = sys.argv[3] + 'k_range.txt'

os.makedirs(os.path.dirname(log_file), exist_ok=True)

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
            
            df1 = torch.Tensor(pd.read_csv(file1, header=None, index_col=0).values)
            df2 = torch.Tensor(pd.read_csv(file2, header=None, index_col=0).values)
            
            #reversed: input2query
            tensor11 = torch.Tensor(df2).cuda()
            tensor22 = torch.Tensor(df1).cuda()
            
            dists = torch.cdist(tensor11, tensor22, p=2)
            # print(dists.shape)
            
            # loop through pairs and calculate rank for each, reversed: input2query
            ranks = []
            for col, row in ground_results:
                row_values = dists[row]
                sorted_indices = torch.argsort(row_values)
                rank = (sorted_indices == col).nonzero().item()
                ranks.append(rank)
                # print(f"Rank of column {col} for row {row}: {rank}")
            
                
                #scores2.append((nocase, nocol, vec, k, rec_qi, rec_iq))
            # break
            # mean = statistics.mean(ranks)
            # median = statistics.median(ranks)
        
            # print("{}:{} has mean {} and median {}".format(nocase, vec, mean, median))
            log = {'vec': vec, 'case':nocase, 'ranks': ranks}
            f.write(json.dumps(log)+'\n')
        #break
    # break
    

