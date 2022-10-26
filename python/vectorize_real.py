#!/usr/bin/env python
import os
import pandas as pd
from vectorization import create_embeddings
import sys
from utils import vectorizers, separators

input_dir = '../../data/real/'
output_dir = '/mnt/data/entity_matching_embeddings/real'
if len(sys.argv) > 1:
    output_dir = sys.argv[1]
log_file = '../logs/vectorization_real.txt'

if __name__ == '__main__':
    
    dirs = separators.keys()
    files = [(dir, file)  for dir in dirs
             for file in os.listdir(input_dir+dir) if 'gt' not in file]
    files = [file for file in files if 'gt' not in file]
    cols = [0, 1, -1]
    
    # print(files)
    
    
    for dir, file in files:
        path = '{}/{}/{}'.format(input_dir, dir, file)
        sep = separators[dir]
        df = pd.read_csv(path, sep=sep, index_col=0)
        print(dir, file, sep, df.shape)
        
        for col in cols:
            colname = df.columns[col]
            data = df[colname]
            
            if data.dtype in ['float64', 'int64']:
                continue
            
            data = data.fillna('')
            
            text = data.tolist()
            #text2 = data.str.split(' ').to_list()
            
            for vectorizer in vectorizers:
                print(vectorizer)

                colname2 = colname.replace('/', '')
                path2 = path.replace(input_dir, output_dir)
                path2 = path2.replace('.csv', f'_{colname2}_{vectorizer}.csv')
                
                log = {}
                log['dir'] = dir
                log['file'] = file
                log['vectorizer'] = vectorizer
                log['column'] = {'name': colname,
                                  'stats': data.apply(len).describe().to_dict()}
                
                embeddings = create_embeddings(text, vectorizer, log, log_file,
                                               path2, df.index)
                print()
        # break
