#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import pandas as pd
from datetime import datetime
from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
import json

print(sys.argv)
print(len(sys.argv))

if len(sys.argv) != 2:
        raise ValueError("Not correct args!")

size = sys.argv[1]
   
    
deli = '|'
#size = '10K'
main_dir = '/home/azeakis/entity_matching/data/big/'
column = 'agg'
#sys.stdout = open(main_dir +size+".txt", 'w')    
    
    
left_df = pd.read_csv(main_dir + "profiles/"+size +".csv", sep=deli, index_col=0)
left_df = left_df.fillna('')
left_df = left_df.apply(lambda x: ' '.join([str(xx) for xx in x]), axis=1)
left_df = left_df.reset_index(drop=False)

left_df.columns = ['id', column]

golden_df = pd.read_csv(main_dir + "ground_truths/" + size+ "duplicates.csv", sep=deli)

l_id = 'ltable_id'
r_id = 'rtable_id'
golden_df.columns = ['ltable_id', 'rtable_id']

corr_sorted = golden_df[golden_df[l_id]<golden_df[r_id]]
inv_sorted = golden_df[golden_df[l_id]<golden_df[r_id]]
inv_sorted = inv_sorted.reindex([r_id, l_id], axis=1)
inv_sorted.columns = [l_id, r_id]
golden_df = pd.merge(corr_sorted, inv_sorted, on=[l_id, r_id], how='outer')
    
cols_to_block = [column]

k = 10
with open('logs.txt', 'a') as out:
        for iteration in (0,5): 
            time_1 = datetime.now()
            
            tuple_embedding_model = AutoEncoderTupleEmbedding()
            topK_vector_pairing_model = ExactTopKVectorPairing(K=k)
            db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
            candidate_set_df = db.block_datasets(left_df, left_df, cols_to_block)
            
            time_2 = datetime.now()
            time_diff = (time_2 - time_1)
            time_diff = time_diff.total_seconds()
            results = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, left_df)
            
            
            log = {'k': k, 'time': time_diff, 'recall': results.get("recall"), 'precision': results.get("pq"),
               'candidates': results.get("candidates"), 'col': column, 'size': size}
            out.write(json.dumps(log)+'\n')  
