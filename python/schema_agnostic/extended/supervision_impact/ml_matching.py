# -*- coding: utf-8 -*-
import pandas as pd
import time
from scipy import spatial, stats
import json
import sys
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def balance_classes(df):
    df_0 = df.loc[df.label == 0]
    df_1 = df.loc[df.label == 1]
    no_1 = df_1.shape[0]
    return pd.concat([df_0.sample(no_1, random_state=1924), df_1])
        
if __name__ == '__main__':     

    
    vectorizers = ['smpnet', 'sdistilroberta', 'sminilm', 'fasttext', 'glove',
                    'bert', 'distilbert', 'roberta', 'xlnet', 'albert', ]

    main_dir = sys.argv[1]
    emb_dir = sys.argv[2]
    
    for balanced in [True, False]:
        bl_log = 'balanced' if balanced else 'imbalanced'
        
        log_file = sys.argv[3] + f'supervision_ml_matching_{bl_log}.txt'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
        with open(log_file, 'w') as out:
            for vectorizer in vectorizers:
                
                datasets = ['abt_buy', 'dirty_amazon_itunes', 'dirty_dblp_acm',
                            'dirty_dblp_scholar', 'dirty_walmart_amazon']
    #            datasets = ['dirty_amazon_itunes']
                
                for nod, dataset in enumerate(datasets):
                    # if nod >= 1:
                    #     continue
                    current_dir = main_dir + dataset
                    case = f'DSM{nod+1}'
                    # print('\n\n' + current_dir)
                    print(vectorizer, case)
                    
                    train = pd.read_csv(current_dir + '/train.csv', na_filter=False)
                    valid = pd.read_csv(current_dir + '/valid.csv', na_filter=False)
                    if balanced:
                        train = balance_classes(train)
                        valid = balance_classes(valid)
                    test = pd.read_csv(current_dir + '/test.csv', na_filter=False)
                    
                    file1 = '{}{}/tableA_aggregate_{}.csv'.format(emb_dir, dataset, vectorizer)
                    file2 = '{}{}/tableB_aggregate_{}.csv'.format(emb_dir, dataset, vectorizer)
                    df1 = pd.read_csv(file1, header=None, index_col=0)
                    df2 = pd.read_csv(file2, header=None, index_col=0)
                    
                    
                    preprocessing_time = time.time()
                    
                    X_train = []
                    for vec1, vec2 in zip(df1.loc[train['left_id']].values, df2.loc[train['right_id']].values):
                        X_train.append( ( 1 - spatial.distance.cosine(vec1, vec2),
                                        1/(1+spatial.distance.euclidean(vec1, vec2)) , 
                                        1/(1+stats.wasserstein_distance(vec1, vec2))))
    
                    for vec1, vec2 in zip(df1.loc[valid['left_id']].values, df2.loc[valid['right_id']].values):
                        X_train.append( ( 1 - spatial.distance.cosine(vec1, vec2),
                                        1/(1+spatial.distance.euclidean(vec1, vec2)) , 
                                        1/(1+stats.wasserstein_distance(vec1, vec2))))
                    
                    y_train = pd.concat([train['label'], valid['label']]).values
                    
                    print(train.shape, valid.shape, len(X_train), len(y_train))
                    
                    X_test = []
                    for vec1, vec2 in zip(df1.loc[test['left_id']].values, df2.loc[test['right_id']].values):
                        X_test.append( ( 1 - spatial.distance.cosine(vec1, vec2),
                                        1/(1+spatial.distance.euclidean(vec1, vec2)) , 
                                        1/(1+stats.wasserstein_distance(vec1, vec2))))
    
                    y_test = test['label']
                    print(test.shape, len(X_test), len(y_test))
                    
                    preprocessing_time = time.time() - preprocessing_time
                    
                    training_time = time.time()
                    # Create an SVM classifier
                    svm_classifier = SVC(kernel='linear')
                    
                    # Define the range of C values to search
                    param_grid = {'C': [0.1, 1, 10, 100],
                                  'kernel': ['linear', 'rbf', 'sigmoid']}
                    
                    # Define the StratifiedKFold cross-validation strategy
                    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Adjust n_splits as needed
    
                    
                    # Perform grid search with cross-validation to find the best C value
                    grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=stratified_cv, n_jobs=10)
                    grid_search.fit(X_train, y_train)
                    
                    # Get the best C value
                    best_C = grid_search.best_params_['C']
                    best_kernel = grid_search.best_params_['kernel']
                    
                    # Train the final model with the best C value on the entire training dataset
                    final_model = SVC(kernel=best_kernel, C=best_C)
                    final_model.fit(X_train, y_train)
                    
                    training_time = time.time() - training_time
                    
                    testing_time = time.time()
                    # Evaluate the model on the testing dataset
                    y_pred = final_model.predict(X_test)
                    testing_time = time.time() - testing_time
                    
                    # Calculate and print F1-score, precision, and recall
                    f1 = f1_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
    
                    # print(y_pred.shape)                
                    print(f"\tF1-score: {f1:.2f}")
                    # print(f"\tPrecision: {precision:.2f}")
                    # print(f"\tRecall: {recall:.2f}")
                    # print(f"\tAccuracy: {accuracy:.2f}")
                    
                    # print(classification_report(y_test, y_pred))
              
                    
                    log = {'vectorizer': vectorizer, 'dataset': case, 
                           'best_C': best_C, 'best_kernel': best_kernel,
                           'f1': f1,  'precision': precision, 'recall': recall, 'accuracy': accuracy,
                           'training_time': training_time, 'testing_time': testing_time,
                           'preprocessing_time': preprocessing_time}
                    out.write(json.dumps(log)+'\n')    
                    out.flush()
                    
                #     break
                # break
