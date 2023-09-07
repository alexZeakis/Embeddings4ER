import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import time
import numpy as np
from utils import vectorizers, cases
import sys


def topk(x, k):
    out = []
    for no, xx in enumerate(x):
        out.append((no, xx))
    out = sorted(out, key=lambda x: x[1])
    return out[:k]


def find_exact_nns(tensor1, tensor2, lpref, rpref, k, batch_size=1000):
    tensor11 = torch.Tensor(tensor1).cuda()
    tensor22 = torch.Tensor(tensor2).cuda()
    
    size1 = tensor11.shape[0]
    size2 = tensor22.shape[0]
    
    num_batches = (size1 + batch_size - 1) // batch_size
    results = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, size1)
        batch1 = tensor11[start_idx:end_idx]
        
        dists = torch.cdist(batch1, tensor22, p=2)
        topk_dists = torch.topk(dists, k, largest=False)
        
        for no, res_list in enumerate(topk_dists.indices):
            for res in res_list:
                left = f'{lpref}{no + start_idx}' if no < (end_idx - start_idx) else f'{lpref}{no + start_idx - size1}'
                right = f'{rpref}{res.item()}'
                results.append((left, right))
    
    results = set(results)
    return results


def find_exact_nns_both(tensor1, tensor2, k, batch_size=1000):
    size = tensor1.shape[0]
    total = torch.cat((tensor1, tensor2), 0).cuda()

    num_batches = (size + batch_size - 1) // batch_size
    results = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, size)
        batch = tensor1[start_idx:end_idx].cuda()

        # Calculate all pairwise distances between elements in the current batch and all elements in the total tensor
        dists = torch.cdist(batch, total, p=2)

        # Sort the distances and get the corresponding indices
        sorted_indices = torch.argsort(dists, dim=1)

        for i in range(batch.shape[0]):
            # Get the sorted indices for the i-th element in the batch
            indices = sorted_indices[i]

            count = 0
            j = 0
            while count < k and j < size:
                idx_in_total = indices[j]
                if start_idx + i != idx_in_total:
                    # Append the (l, r) or (r, l) pair to the results
                    left = f'l_{start_idx + i}' if start_idx + i < size else f'r_{start_idx + i - size}'
                    right = f'l_{idx_in_total}' if idx_in_total < size else f'r_{idx_in_total - size}'
                    results.append((left, right))
                    count += 1
                j += 1

    return set(results)



def evaluate(true, preds):
    prec = len(true & preds) / len(preds)
    rec = len(true & preds) / len(true)
    if prec==rec==0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec+rec)
    return prec, rec, f1



input_dir = sys.argv[1]
emb_dir = sys.argv[2]
log_file = sys.argv[3] + 'indexing.csv'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

ks = [1, 5, 10]

scores2 = []


for nocase, (data1, data2, ground_file, sep, dir, cols) in enumerate(cases):
    nocase = f'D{nocase+1}'
    print()
    print(nocase)
    #print(noc, data1, data2, ground_file, sep, dir, cols)
    
    ground_file = '{}{}/{}.csv'.format(input_dir, dir, ground_file)
    ground_df = pd.read_csv(ground_file, sep=sep)
    ground_results = set(ground_df.apply(lambda x: (f'l_{x[0]}', f'r_{x[1]}'), axis=1).values)
    
    for nocol, (col1, col2) in enumerate(cols):
        if nocol != 2:
            continue
        for vec in vectorizers:
            file1 = '{}{}/{}_{}_{}.csv'.format(emb_dir, dir, data1, col1, vec)
            file2 = '{}{}/{}_{}_{}.csv'.format(emb_dir, dir, data2, col2, vec)
            
            df1 = pd.read_csv(file1, header=None, index_col=0).values
            df2 = pd.read_csv(file2, header=None, index_col=0).values
            df1 = torch.Tensor(df1)
            df2 = torch.Tensor(df2)
            
            for k in ks:
                # print('\t{} {} {}\r'.format(nocol, vec, k), end='')
                
                # #exact - Left->Right
                # t1 = time()
                # results = find_exact_nns(df1, df2, 'l_', 'r_', k)
                # t2 = time()
                # prec, rec, f1 = evaluate(ground_results, results)
                # scores2.append((nocase, nocol, vec, k, 'left', 'exact', prec, rec, f1, t2-t1))
                
                # #exact - Right->Left
                # t1 = time()
                # results = find_exact_nns(df2, df1, 'r_', 'l_', k)
                # t2 = time()
                # results = set([(y,x) for x,y in results]) #reverse the input to query results to become (q_id, in_id)
                # prec, rec, f1 = evaluate(ground_results, results)
                # scores2.append((nocase, nocol, vec, k, 'right', 'exact', prec, rec, f1, t2-t1))

                
                t1 = time()
                # results = find_exact_nns_both(df1, df2, k+1)
                # exact - Left->Right
                results1 = find_exact_nns(df1, df2, 'l_', 'r_', k)
                #exact - Right->Left
                results2 = find_exact_nns(df2, df1, 'r_', 'l_', k)
                results2 = set([(y,x) for x,y in results2]) #reverse the input to query results to become (q_id, in_id)
                # exact - Both->Both
                results = results1 | results2
                t2 = time()
                
                prec, rec, f1 = evaluate(ground_results, results1)
                scores2.append((nocase, nocol, vec, k, 'right', 'exact', prec, rec, f1, t2-t1))
                
                prec, rec, f1 = evaluate(ground_results, results2)
                scores2.append((nocase, nocol, vec, k, 'left', 'exact', prec, rec, f1, t2-t1))
                
                prec, rec, f1 = evaluate(ground_results, results)
                scores2.append((nocase, nocol, vec, k, 'both', 'exact', prec, rec, f1, t2-t1))

    #             break
    #         break
    #     break
    # break
                
results = pd.DataFrame(scores2, columns=['Case', 'Columns', 'Vectorizer', 'k', 'Direction',
                                          'Exact', 'Precision', 'Recall', 'F1', 'Time'])    
results.to_csv(log_file, header=True, index=True)

