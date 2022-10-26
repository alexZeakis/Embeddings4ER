import pandas as pd
import torch
from scipy.spatial.distance import cdist
from time import time
import faiss
import hnswlib
import numpy as np
import sys
from utils import vectorizers


# ## Nearest-Neighbor Search

def topk(x, k):
    out = []
    for no, xx in enumerate(x):
        out.append((no, xx))
    out = sorted(out, key=lambda x: x[1])
    return out[:k]

def find_exact_nns(tensor1, tensor2, k, offset, gpu=False):
    if gpu:
        tensor11 = torch.Tensor(tensor1)
        tensor22 = torch.Tensor(tensor2)
        
        dists = torch.cdist(tensor11, tensor22, p=2)
        
        topk_dists = torch.topk(dists, k+1, largest=False)
        results = []
        for no, res_list in enumerate(topk_dists.indices):
            for res in res_list:
                #results.append((no+offset, res.item()+offset))
                nores = res.item()
                if no+offset == nores:
                    continue
                # if no+offset < nores:
                    # results.append((no+offset, nores))
                results.append(tuple(sorted((no+offset, nores))))
        #results = set(results)
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
        distances, labels = index.search(tensor11, k=k+1)

        results = []
        for no, label_list in enumerate(labels):
            for res in label_list:
                if no == res:
                    continue
                elif no < res:
                    left, right = no, res
                else:
                    left, right = res, no
                results.append((left, right))
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


# # Start NNS Euclidean - Synthetic

emb_dir = '/mnt/data/entity_matching_embeddings/big'
if len(sys.argv) > 1:
    emb_dir = sys.argv[1]

#ks = [1, 5, 10]
ks = [10]
gpu = True

scores2 = []

files = ['10K.csv', '50K.csv', '100K.csv', '200K.csv', '300K.csv', '1M.csv', '2M.csv']
nocol, col, sep = 2, 'aggregated', "|"

batch_size = 5000

i=0
with open('../logs/blocking_recall_euclidean_big.csv', 'w') as o:
        o.write('Case,Columns,Vectorizer,k,Direction,Exact,Recall,Precision,Time\n')
        for nocase, file in enumerate(files):
            
            name = file.split('.')[0]
            print()
            print(name)
            
            
            ground_file = './../../data/big/ground_truths/{}duplicates.csv'.format(name)
            ground_df = pd.read_csv(ground_file, sep=sep)
            ground_results = set(ground_df.apply(lambda x: (x[0], x[1]), axis=1).values)
            
            for vec in vectorizers:
                print('\t{}\r'.format(vec), end='')
                file = '{}/{}_{}_{}.csv'.format(emb_dir, name, col, vec)
                df = pd.read_csv(file, header=None, index_col=0).values
                
                for k in ks:
                    
                    #exact - NNS
                    t1 = time()
                    #print()
                    
                    no_batches = df.shape[0] // batch_size
                    offset = 0
                    results = []
                    for i in range(no_batches):
                        temp_df = df[offset: offset + batch_size]
                        #rest_df = df[offset:]
                        temp_results = find_exact_nns(temp_df, df, k, offset, gpu)
                        #print(i, len(temp_results))
                        offset += batch_size
                        results += temp_results
                    
                    results = set(results)
                    #print(len(results))
                    
                    t2 = time()
                    

                    recall = calc_recall(ground_results, results)
                    precision = calc_precision(ground_results, results)
                    #scores2.append((nocase, nocol, vec, k, 'i2q', 'exact', recall, precision, t2-t1))
                    o.write('{},{},{},{},{},{},{},{},{},{}\n'.format(i, nocase, nocol, vec, k, 'i2q', 'exact', recall, precision, t2-t1))

                    #approx - NNS
                    t1 = time()
                    results = find_approx_nns(df, df, k, gpu)
                    t2 = time()
                    recall = calc_recall(ground_results, results)
                    precision = calc_precision(ground_results, results)
                    #scores2.append((nocase, nocol, vec, k, 'i2q', 'approx', recall, precision, t2-t1))  
          
                    o.write('{},{},{},{},{},{},{},{},{},{}\n'.format(i, nocase, nocol, vec, k, 'i2q', 'approx', recall, precision, t2-t1))
                    i += 1