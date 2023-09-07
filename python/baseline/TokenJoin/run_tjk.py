import os
import pandas as pd
from time import time
import json
import sys
from pytokenjoin.jaccard.jaccard_knn import JaccardTokenJoin as jtk

def evaluate(true, preds):
    prec = len(true & preds) / len(preds)
    rec = len(true & preds) / len(true)
    if prec==rec==0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec+rec)
    return prec, rec, f1


cases = [
        ('rest1', 'rest2', 'gt', "|", 'D1(rest)'),
        ('abt', 'buy', 'gt', "|", 'D2(abt-buy)'),
        ('amazon', 'gp', 'gt', "#", 'D3(amazon-gp)'),
        ('dblp', 'acm', 'gt', "%", 'D4(dblp-acm)'), 
        ('imdb', 'tvdb', 'gtImTv', "|", 'D5_D6_D7(imdb-tmdb)'),
        ('tmdb', 'tvdb', 'gtTmTv', "|", 'D5_D6_D7(imdb-tmdb)'), 
        ('imdb', 'tmdb', 'gtImTm', "|", 'D5_D6_D7(imdb-tmdb)'), 
        ('walmart', 'amazon', 'gt', "|", 'D8(walmart-amazon)'), 
        ('dblp', 'scholar', 'gt', ">", 'D9(dblp-scholar)'), 
        ('imdb', 'dbpedia', 'gtImDb', "|", 'D10(movies)'), 
        ]


input_dir = sys.argv[1]
log_file = sys.argv[2] + 'TokenJoin.txt'

os.makedirs(os.path.dirname(log_file), exist_ok=True)

if __name__ == '__main__':
    
    cid, tid = 'id', 'aggregate value'
    stats = []
    with open(log_file, 'a') as out:
        for noc, (file1, file2, ground, sep, dir) in enumerate(cases):
            noc = f'D{noc+1}'
            df1 = pd.read_csv('{}/{}/{}.csv'.format(input_dir, dir, file1), sep=sep)
            df2 = pd.read_csv('{}/{}/{}.csv'.format(input_dir, dir, file2), sep=sep)
            gt = pd.read_csv('{}/{}/{}.csv'.format(input_dir, dir, ground), sep=sep)
            
            df1[tid] = df1[tid].fillna('')
            df2[tid] = df2[tid].fillna('')
            
            df1[tid] = df1[tid].apply(lambda x: list(set(x.split(' '))))
            df2[tid] = df2[tid].apply(lambda x: list(set(x.split(' '))))
            
            true = set(gt[['D1', 'D2']].apply(lambda x: (x[0], x[1]), axis=1).tolist())
            
#            for k in [1, 5]:
            for k in [10]:
                t1 = time()
                output_df = jtk().tokenjoin_foreign(df2, df1, cid, cid, tid, tid, k=k)
                t2 = time()
                # preds = set(output_df[['l_id', 'r_id']].apply(lambda x: (x[0], x[1]), axis=1).tolist())
                preds = set(output_df[['l_id', 'r_id']].apply(lambda x: (x[1], x[0]), axis=1).tolist())
                
                prec, rec, f1 = evaluate(true, preds)
                out.write(json.dumps({'case': noc, 'k': k, 'time': t2-t1,
                                      'prec': prec, 'rec': rec, 'f1':f1})+"\n")
                out.flush()
            # break
