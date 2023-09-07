import os
import pandas as pd
from time import time
import json
from pyjedai.datamodel import Data
from pyjedai.joins import TopKJoin
import sys

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

if __name__ == '__main__':
    
    input_dir = sys.argv[1]
    log_file = sys.argv[2] + 'JedAI.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    cid, tid = 'id', 'aggregate value'
    stats = []
    with open(log_file, 'a') as out:
        for noc, (file1, file2, ground, sep, dir) in enumerate(cases):
            noc = f'D{noc+1}'
            ground_truth=pd.read_csv('{}/{}/{}.csv'.format(input_dir, dir, ground), sep=sep)
            ground_truth['temp'] = ground_truth['D1']
            ground_truth['D1'] = ground_truth['D2']
            ground_truth['D2'] = ground_truth['temp']
            ground_truth = ground_truth[['D1', 'D2']]
        
            data = Data(
                    dataset_1=pd.read_csv('{}/{}/{}.csv'.format(input_dir, dir, file2), sep=sep, na_filter=False).astype(str),
                    attributes_1=[tid],
                    id_column_name_1=cid,
                    dataset_2=pd.read_csv('{}/{}/{}.csv'.format(input_dir, dir, file1), sep=sep, na_filter=False).astype(str),
                    attributes_2=[tid],
                    id_column_name_2=cid,
                    ground_truth=ground_truth,
                )

#            for k in [1, 5]:
            for k in [1, 5, 10]:
                t1 = time()
                join = TopKJoin(K = k, metric = 'cosine', tokenization = 'qgrams_multiset', qgrams = 5)
                g = join.fit(data)
                report = join.evaluate(g)
                t2 = time()
                
                out.write(json.dumps({'case': noc, 'k': k, 'time': t2-t1,
                                      'prec': report['Precision %'] / 100, 'rec': report['Recall %'] / 100, 'f1':report['F1 %'] / 100})+"\n")
                out.flush()
            # break
