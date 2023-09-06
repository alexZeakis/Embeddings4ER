#!/usr/bin/env python
import pandas as pd
import torch
from time import time
from utils import cases, cosine_similarity
import sys
import os

vectorizers = ['st5']


gpu = True
cosine = False

if cosine:
   sim = 'cosine'
else:
   sim = 'euclidean'

data_dir = sys.argv[1]
emb_dir = sys.argv[2]
log_file = sys.argv[3] + f'matching_unsupervised_{sim}_block.csv'

scores2 = []

os.makedirs(os.path.dirname(log_file), exist_ok=True)
for nocase, (data1, data2, ground_file, sep, dir, cols) in enumerate(cases):
    # if nocase >0:
    #     break
    nocase = f'D{nocase+1}'
    print()
    print(nocase)
    #print(noc, data1, data2, ground_file, sep, dir, cols)
    
    ground_file = '{}/{}/{}.csv'.format(data_dir, dir, ground_file)
    ground_df = pd.read_csv(ground_file, sep=sep)
    # ground_results = set(ground_df.apply(lambda x: (f'r_{x[0]}', f's_{x[1]}'), axis=1).values)
    ground_results = set(ground_df.apply(lambda x: (x[0], x[1]), axis=1).values)
    
    for nocol, (col1, col2) in enumerate(cols):
        if nocol!=2:
            continue
        for vec in vectorizers:
            print('\t{} {}\r'.format(nocol, vec), end='')
            torch.cuda.empty_cache()
            #print()
            file1 = '{}{}/{}_{}_{}.csv'.format(emb_dir, dir, data1, col1, vec)
            file2 = '{}{}/{}_{}_{}.csv'.format(emb_dir, dir, data2, col2, vec)
            
            df1 = pd.read_csv(file1, header=None, index_col=0).values
            df2 = pd.read_csv(file2, header=None, index_col=0).values
                
            #cdists
            dist_time = time()
            
            df1 = torch.Tensor(df1).cuda()
            df2 = torch.Tensor(df2).cuda()
            
            #df1 = torch.Tensor(df1)
            #df2 = torch.Tensor(df2)
            
            if cosine:
                    dists = cosine_similarity(df1, df2)
            else:
                    dists = torch.cdist(df1, df2, p=2)
                    # dists = 1 / (1+dists)
                    dists.add_(1.0).pow_(-1)

            vals2, inds2 = dists.topk(k=10, largest=True)
            edges = []
            for no in range(vals2.shape[0]):
                for no2 in range(vals2[no].shape[0]):
                    edges.append((no, inds2[no][no2].item(), vals2[no][no2].item()))
            edges = sorted(edges, key=lambda x: x[2], reverse=True)
                
            # values, indices = dists.flatten().sort(descending=True)
            div = dists.shape[1]
            
            df1_shape = df1.shape[0]
            df2_shape = df2.shape[0]
            
            del df1
            del df2
            min_collection = min(df1_shape, df2_shape)
            dist_time = time() - dist_time
            
            matching_time = time()
            results = []
            # l_matched = set()
            # r_matched = set()
            l_matched = [False for _ in range(df1_shape)]
            r_matched = [False for _ in range(df2_shape)]
            
            delta = 0.95
            
            for noind, (i, j, dist) in enumerate(edges):
                
                while dist < delta:
                    results2 = set(results)
                    matching_time2 = time() - matching_time
        
                    true_positives = len(ground_results & results2)
                    if true_positives > 0:
                        recall =  true_positives / len(ground_results)
                        precision =  true_positives / len(results2)
                        f1 = 2 * (precision*recall) / (precision + recall)
                    else:
                        recall, precision, f1 = 0, 0, 0
                   
                    scores2.append((nocase, nocol, vec, recall, precision, f1, dist_time, matching_time2, len(results2), delta)) 
                    delta -= 0.05
                
                # if noind % 1000000 == 0:
                #     print('{:,} / {:,} -> {:,} / {:,}'.format(noind, len(dists), len(results), min_collection))
                
                # if i in l_matched or j in r_matched:
                if l_matched[i] or r_matched[j]:
                    continue
                results.append((i, j))
                # l_matched.add(i)
                # r_matched.add(j)
                l_matched[i] = True
                r_matched[j] = True
                
                if len(results) == min_collection:
                    break
                
            
            results = set(results)
            matching_time = time() - matching_time


            #recall
            recall_time = time()
            true_positives = len(ground_results & results)
            if true_positives > 0:
                recall =  true_positives / len(ground_results)
                precision =  true_positives / len(results)
                f1 = 2 * (precision*recall) / (precision + recall)
            else:
                recall, precision, f1 = 0, 0, 0
            recall_time = time() - recall_time  
            
            del dists

            # print((nocase, nocol, vec, recall, precision, f1, matching_time))
            #scores2.append((nocase, nocol, vec, q, recall, precision, f1, matching_time))
            scores2.append((nocase, nocol, vec, recall, precision, f1, dist_time, matching_time, len(results), delta))
            #break
        #break
    #break

results = pd.DataFrame(scores2, columns=['Case', 'Columns', 'Vectorizer', 'Recall', 'Precision', 'F1', 'Blocking Time', 'Matching Time',
                                         '#Results', 'Delta'
                                         ])
# #results = pd.DataFrame(scores2, columns=['Case', 'Columns', 'Vectorizer', 'Limit', 'Recall', 'Precision', 'F1', 'Matching Time'])    

results.to_csv(log_file, header=True, index=False)
