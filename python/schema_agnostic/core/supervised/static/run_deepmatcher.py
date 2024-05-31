import deepmatcher as dm
import torch
import os
print(torch.cuda.is_available())
import sys
from time import time
import json

model_name=sys.argv[1]
model_dir=sys.argv[2]
data_dir=sys.argv[3]
log_dir=sys.argv[4]


models = { 'fasttext': 'fasttext.en.bin', 'glove': 'glove.6B.300d', 
           'word2vec': 'word2vec-google-news-300.gz'
           }

print(model_name, data_dir)

dm.data.reset_vector_cache()

if os.path.exists(data_dir+'/cacheddata.pth'):
    os.remove(data_dir+'/cacheddata.pth')

print(model_dir+model_name+'/')
train, validation, test = dm.data.process(
    path=data_dir,
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    embeddings=models[model_name],
    embeddings_cache_path= model_dir+model_name+'/',
    auto_rebuild_cache=False, pca = False
)

model = dm.MatchingModel()
train_time = time()
model.run_train(train, validation, best_save_path='best_model.pth', device="cpu")
train_time = time() - train_time
test_time = time()
f1 = model.run_eval(test, device="cpu")
test_time = time() - test_time

result = {'model_type': model_name, 'data_name': data_dir.split('/')[-1], 
          'training_time': train_time, 'testing_time': test_time, 'f1': f1.item()}

log_file = log_dir + 'matching_supervised_static.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
with open(log_file, 'a') as f:
   f.write(json.dumps(result)+'\n')
