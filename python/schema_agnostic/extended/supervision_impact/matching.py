# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 03:35:44 2022

@author: G_A.Papadakis
"""

import numpy as np
import pandas as pd
import time
from scipy import spatial, stats
import json
import sys
import os

        
if __name__ == '__main__':     

    
    vectorizers = ['smpnet', 'sdistilroberta', 'sminilm', 'fasttext', 'glove',
                   'bert', 'distilbert', 'roberta', 'xlnet', 'albert', ]

    main_dir = sys.argv[1]
    emb_dir = sys.argv[2]
    log_file = sys.argv[3] + 'supervision_matching.txt'

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'w') as out:
        for vectorizer in vectorizers:
            
            similarities = ['CS', 'ES', 'WS']
            datasets = ['abt_buy', 'dirty_amazon_itunes', 'dirty_dblp_acm',
                        'dirty_dblp_scholar', 'dirty_walmart_amazon']
#            datasets = ['dirty_amazon_itunes']
            
            for nod, dataset in enumerate(datasets):
                current_dir = main_dir + dataset
                case = f'DSM{nod+1}'
                # print('\n\n' + current_dir)
                print(vectorizer, case)
                
                train = pd.read_csv(current_dir + '/train.csv', na_filter=False)
                valid = pd.read_csv(current_dir + '/valid.csv', na_filter=False)
                test = pd.read_csv(current_dir + '/test.csv', na_filter=False)
                
                file1 = '{}{}/tableA_aggregate_{}.csv'.format(emb_dir, dataset, vectorizer)
                file2 = '{}{}/tableB_aggregate_{}.csv'.format(emb_dir, dataset, vectorizer)
                df1 = pd.read_csv(file1, header=None, index_col=0)
                df2 = pd.read_csv(file2, header=None, index_col=0)
                
                
                time_1 = time.time()
                
                best_thresholds = []
                
                # train['CS'], train['ES'], train['WS'] = zip(*train.apply(lambda row : get_similarities(model, d1_cols, d2_cols, row), axis = 1))
                train_cs, train_es, train_ws = [], [], []
                for vec1, vec2 in zip(df1.loc[train['left_id']].values, df2.loc[train['right_id']].values):
                    train_cs.append( 1 - spatial.distance.cosine(vec1, vec2))
                    train_es.append( 1/(1+spatial.distance.euclidean(vec1, vec2)))
                    train_ws.append( 1/(1+stats.wasserstein_distance(vec1, vec2)))
                train['CS'] = train_ws
                train['ES'] = train_es
                train['WS'] = train_ws
                
                for measure_id in range(len(similarities)):
                    best_F1, bestThr = -1, -1
                    for threshold in np.arange(0.01, 1.00, 0.01):                
                        train['pred_label'] = threshold <= train[similarities[measure_id]]
                        
                        tp = len(train[(train['label'] == 1) & train['pred_label']])
                        fp = len(train[(train['label'] == 0) & train['pred_label']])
                        fn = len(train[(train['label'] == 1) & (train['pred_label'] == False)])
                        
                        precision = tp / (tp + fp) if tp + fp > 0 else 0
                        recall = tp / (tp + fn) if tp + fn > 0 else 0
                        if ((0 < precision) & (0 < recall)):
                            f1 = 2 * precision * recall / (precision + recall)
                            if (best_F1 < f1):
                                best_F1 = f1
                                bestThr = threshold
                    #print(best_F1, bestThr)
                    best_thresholds.append(bestThr)
                
                best_F1, best_measure_id = -1, -1
                # valid['CS'], valid['ES'], valid['WS'] = zip(*valid.apply(lambda row : get_similarities(model, d1_cols, d2_cols, row), axis = 1))
                
                valid_cs, valid_es, valid_ws = [], [], []
                for vec1, vec2 in zip(df1.loc[valid['left_id']].values, df2.loc[valid['right_id']].values):
                    valid_cs.append( 1 - spatial.distance.cosine(vec1, vec2))
                    valid_es.append( 1/(1+spatial.distance.euclidean(vec1, vec2)))
                    valid_ws.append( 1/(1+stats.wasserstein_distance(vec1, vec2)))
                valid['CS'] = valid_cs
                valid['ES'] = valid_es
                valid['WS'] = valid_ws
                
                for measure_id in range(len(similarities)):            
                    valid['pred_label'] = best_thresholds[measure_id] <= valid[similarities[measure_id]]
                    
                    tp = len(valid[(valid['label'] == 1) & valid['pred_label']])
                    fp = len(valid[(valid['label'] == 0) & valid['pred_label']])
                    fn = len(valid[(valid['label'] == 1) & (valid['pred_label'] == False)])
                    
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    if ((0 < precision) & (0 < recall)):
                        f1 = 2 * precision * recall / (precision + recall)
                        #print(measure_id, f1)
                        if (best_F1 < f1):
                            best_F1 = f1
                            best_measure_id = measure_id
                
                time_2 = time.time()
                    
                # test['sim'] = test.apply(lambda row : get_similarity(model, d1_cols, d2_cols, best_measure_id, row), axis = 1)
                test_sim = []
                for vec1, vec2 in zip(df1.loc[test['left_id']].values, df2.loc[test['right_id']].values):
                    if best_measure_id == 0:
                        test_sim.append( 1 - spatial.distance.cosine(vec1, vec2))
                    elif best_measure_id == 1:
                        test_sim.append( 1/(1+spatial.distance.euclidean(vec1, vec2)))
                    elif best_measure_id == 2:
                        test_sim.append( 1/(1+stats.wasserstein_distance(vec1, vec2)))
                test['sim'] = test_sim
            
            
                test['pred_label'] = best_thresholds[best_measure_id] <= test['sim']
                tp = len(test[(test['label'] == 1) & test['pred_label']])
                fp = len(test[(test['label'] == 0) & test['pred_label']])
                fn = len(test[(test['label'] == 1) & (test['pred_label'] == False)])
            
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                
                time_3 = time.time()
                
                f1 = 2 * precision * recall / (precision + recall)
                training_time = time_2-time_1
                testing_time = time_3-time_2
                
                #print('Validation results', best_F1, similarities[best_measure_id], best_thresholds[best_measure_id])
                #print('Final F1', 2 * precision * recall / (precision + recall))
                #print('Training time (sec)', time_2-time_1)
                #print('Testing time (sec)', time_3-time_2)                
                
                log = {'vectorizer': vectorizer, 'dataset': case, 'best_f1': best_F1,
                       'best_similarity': similarities[best_measure_id], 
                       'best_threshold': best_thresholds[best_measure_id],
                       'f1': f1,  'precision': precision, 'recall': recall,
                       'training_time': training_time, 'testing_time': testing_time}
                out.write(json.dumps(log)+'\n')    
                out.flush()
                
                # break
            # break
