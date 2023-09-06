#!/usr/bin/env python
import os
import pandas as pd
from vectorization import create_embeddings
from utils import vectorizers
import sys

separators = {
    'D1(rest)': "|",
    'D2(abt-buy)': "|",
    'D3(amazon-gp)': "#",
    'D4(dblp-acm)': "%",
    'D5_D6_D7(imdb-tmdb)': "|",
    'D8(walmart-amazon)': "|",
    'D9(dblp-scholar)': ">",
    'D10(movies)': "|"
    }


input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_file = sys.argv[3] + 'vectorization_real.txt'

if __name__ == '__main__':
    
    dirs = separators.keys()
    files = [(dir, file)  for dir in dirs
             for file in os.listdir(input_dir+dir) if 'gt' not in file]
    files = [file for file in files if 'gt' not in file]
    cols = [-1]
    
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
                
                os.makedirs(os.path.dirname(path2), exist_ok=True)
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
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
