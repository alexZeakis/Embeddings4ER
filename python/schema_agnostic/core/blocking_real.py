import os
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from time import time
import faiss
import hnswlib
import numpy as np
import sys
from utils import cases, vectorizers



def topk(x, k):
    out = []
    for no, xx in enumerate(x):
        out.append((no, xx))
    out = sorted(out, key=lambda x: x[1])
    return out[:k]

def find_exact_nns(tensor1, tensor2, k, gpu=False):
    if gpu:
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
    else:
        dists = cdist(tensor1, tensor2)
        results = []
        for no, dist_list in enumerate(dists):
            for res in topk(dist_list, k):
                results.append((no, res[0]))
        results = set(results)
        return results
    
def find_approx_nns(tensor1, tensor2, k, gpu=False):
    num_elements, dim = tensor2.shape

    if gpu:
        tensor11 = np.float32(tensor1.copy(order='C'))
        tensor22 = np.float32(tensor2.copy(order='C'))

        index = faiss.IndexHNSWFlat(dim, 16)
        index.hnsw.efConstruction = 200
        index.add(tensor22)  # build the index
        index.hnsw.efSearch = 50
        distances, labels = index.search(tensor11, k=k)

        results = []
        for no, label_list in enumerate(labels):
            for res in label_list:
                results.append((no, res))
        results = set(results)
        return results
    else:
        p = hnswlib.Index(space = 'l2', dim = dim)
        p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
        p.add_items(tensor2)
        p.set_ef(50)
        labels, distances = p.knn_query(tensor1, k = k)
    
        results = []
        for no, label_list in enumerate(labels):
            for res in label_list:
                results.append((no, res))
        results = set(results)
        return results


def calc_recall(true, preds):
    return len(true & preds) / len(true)

def calc_precision(true, preds):
    return len(true & preds) / len(preds)


# # Start NNS euclidean - Real


data_dir = sys.argv[1]
emb_dir = sys.argv[2]
log_file = sys.argv[3] + 'blocking_euclidean_real.csv'
ks = [1, 5, 10]

gpu = True

scores2 = []


for nocase, (data1, data2, ground_file, sep, dir, cols) in enumerate(cases):
    nocase = f'D{nocase+1}'
    print()
    print(nocase)
    #print(noc, data1, data2, ground_file, sep, dir, cols)
    
    ground_file = '{}/{}/{}.csv'.format(data_dir, dir, ground_file)
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
                
                '''
                #exact - query2input
                t1 = time()
                results = find_exact_nns(df1, df2, k, gpu)
                t2 = time()
                recall = calc_recall(ground_results, results)
                precision = calc_precision(ground_results, results)
                scores2.append((nocase, nocol, vec, k, 'q2i', 'exact', recall, precision, t2-t1))
                '''
                                
                #exact - input2query
                t1 = time()
                results = find_exact_nns(df2, df1, k, gpu)
                t2 = time()
                results = set([(y,x) for x,y in results]) #reverse the input to query results to become (q_id, in_id)
                recall = calc_recall(ground_results, results)
                precision = calc_precision(ground_results, results)
                scores2.append((nocase, nocol, vec, k, 'i2q', 'exact', recall, precision, t2-t1))
                
                '''                
                #approx - query2input
                t1 = time()
                results = find_approx_nns(df1, df2, k, gpu)
                t2 = time()
                recall = calc_recall(ground_results, results)
                precision = calc_precision(ground_results, results)
                scores2.append((nocase, nocol, vec, k, 'q2i', 'approx', recall, precision, t2-t1))  
                
                #approx - input2query
                t1 = time()
                results = find_approx_nns(df2, df1, k, gpu)
                t2 = time()
                results = set([(y,x) for x,y in results]) #reverse the input to query results to become (q_id, in_id)
                recall = calc_recall(ground_results, results)
                precision = calc_precision(ground_results, results)
                scores2.append((nocase, nocol, vec, k, 'i2q', 'approx', recall, precision, t2-t1))
                '''
                #scores2.append((nocase, nocol, vec, k, rec_qi, rec_iq))
            #break
        #break
    break
    
os.makedirs(os.path.dirname(log_file), exist_ok=True)
results = pd.DataFrame(scores2, columns=['Case', 'Columns', 'Vectorizer', 'k', 'Direction',
                                         'Exact', 'Recall', 'Precision', 'Time'])    
results.to_csv(log_file, header=True, index=True)
results

