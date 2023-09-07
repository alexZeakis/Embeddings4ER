#!/usr/bin/env python
import os
import pandas as pd
from vectorization import create_embeddings
from itertools import product
import sys

vectorizers = ['glove',
               'bert', 'distilbert', 'roberta', 'xlnet', 'albert', 
               'smpnet', 'sdistilroberta', 'sminilm']  #word2vec and s-gtr-t5 not supported in supervised

input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_file = sys.argv[3] + 'supervision_vectorization.txt'
static_dir = sys.argv[4]

if __name__ == '__main__':
    
    dirs = ['dirty_amazon_itunes', 'abt_buy', 'dirty_walmart_amazon', 'dirty_dblp_acm', 'dirty_dblp_scholar']
    #dirs = ['abt_buy', 'dirty_walmart_amazon', 'dirty_dblp_acm', 'dirty_dblp_scholar']
    files = ['tableA.csv', 'tableB.csv']
    
    # print(files)

    for dir, file in product(dirs, files):
    
        path = '{}{}/{}'.format(input_dir, dir, file)
        df = pd.read_csv(path, index_col=0)
        print(dir, file, df.shape)
        

        #if data.dtype in ['float64', 'int64']:
        #    continue
        
        df = df.fillna('')
        data = df.apply(lambda x: ' '.join([str(col) for col in x]), axis=1)
        
        text = data.tolist()
        #text2 = data.str.split(' ').to_list()
        
        for vectorizer in vectorizers:
            print(vectorizer)

            path2 = path.replace(input_dir, output_dir)
            path2 = path2.replace('.csv', f'_aggregate_{vectorizer}.csv')

            os.makedirs(os.path.dirname(path2), exist_ok=True)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            log = {}
            log['dir'] = dir
            log['file'] = file
            log['vectorizer'] = vectorizer
            log['column'] = {'name': 'aggregate',
                              'stats': data.apply(len).describe().to_dict()}
            
            embeddings = create_embeddings(text, vectorizer, log, log_file,
                                           path2, df.index, static_dir)
            print()
        # break
