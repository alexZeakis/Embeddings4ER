import pandas as pd
from vectorization import create_embeddings
import sys
import os
from utils import vectorizers


input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_file = sys.argv[3] + 'vectorization_synthetic.txt'    
static_dir = sys.argv[4]    

files = ['10K.csv', '50K.csv', '100K.csv', '200K.csv', '300K.csv', '1M.csv', '2M.csv']

for file in files:
    df = pd.read_csv(input_dir+file, sep="|", index_col=0)
    df = df.fillna('')
    df = df.apply(lambda x: ' '.join([str(xx) for xx in x]), axis=1)
    print(file, df.shape)
    
    for vectorizer in vectorizers:
        print(vectorizer)

        text = df.tolist()
        
        path2 = output_dir+file
        path2 = path2.replace('.csv', f'_aggregated_{vectorizer}.csv')
        
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        log = {}
        #log['dir'] = dir
        log['file'] = file
        log['vectorizer'] = vectorizer
        log['column'] = {'name': 'aggregated',
                          'stats': df.apply(len).describe().to_dict()}
                
        embeddings = create_embeddings(text, vectorizer, log, log_file,
                                       path2, df.index, static_dir)
        print()
        #break
    #break
