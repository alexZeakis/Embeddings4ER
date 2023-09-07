import time
import pandas as pd
from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
from utils import cases
import json
import sys
import os

main_dir = sys.argv[1]
log_dir = sys.argv[2]
#main_dir = './data/'

log_file = log_dir+'DeepBlocker.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

with open(log_file, 'w') as out:
    for noc, case in enumerate(cases):
        print(case)
        noc = f'D{noc+1}'

        # cols_to_block = case[5]
        for col_to_block in ['aggregate value']:

            left_df = pd.read_csv('{}{}/{}.csv'.format(main_dir, case[4], case[0]), sep=case[3])
            left_df[col_to_block] = left_df[col_to_block].astype(str)
            right_df = pd.read_csv('{}{}/{}.csv'.format(main_dir, case[4], case[1]), sep=case[3])
            right_df[col_to_block] = right_df[col_to_block].astype(str)  
            golden_df = pd.read_csv('{}{}/{}.csv'.format(main_dir, case[4], case[2]), sep=case[3])
            golden_df.columns = ['ltable_id', 'rtable_id']
            
            
            iterations = 5

            for k in [1, 5, 10]:
                avRecall, avPrecision, avCandidates, avtime = 0, 0, 0, 0
                for iteration in range(0, iterations):
                    start_time = time.time()
                    
                    tuple_embedding_model = AutoEncoderTupleEmbedding()
                    topK_vector_pairing_model = ExactTopKVectorPairing(K=k)
                    db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
                    # candidate_set_df = db.block_datasets(left_df, right_df, [col_to_block])
                    candidate_set_df = db.block_datasets(right_df, left_df, [col_to_block]) # SWITCH
                    
                    candidate_set_df.columns =  ['rtable_id', 'ltable_id']   # SWITCH
                    
                    avtime += time.time() - start_time
                    
                    # results = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
                    results = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, right_df, left_df) # SWITCH
                    avRecall += results.get("recall")
                    avPrecision += results.get("pq")
                    avCandidates += results.get("candidates")
                    
                log = {'k': k, 'time': avtime/iterations, 'recall': avRecall/iterations, 'precision': avPrecision/iterations,
                       'candidates': avCandidates/iterations, 'col': col_to_block, 'case': noc}
                out.write(json.dumps(log)+'\n')    
                out.flush()
                
        #         break
        #     break
        # break
