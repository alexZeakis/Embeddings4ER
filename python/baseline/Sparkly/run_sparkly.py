import os
import pandas as pd
from time import time
import json
import sys
sys.path.append('.')
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparkly.index import IndexConfig, LuceneIndex
from sparkly.search import Searcher
from pathlib import Path

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

def run_sparkly(index, query, gt, sep, cid, tid, limit):
    # path to the test data
    # data_path = Path('./examples/data/abt_buy/').absolute()
    # # table to be indexed
    # table_a_path = data_path / 'table_a.csv'
    # # table for searching
    # table_b_path = data_path / 'table_b.csv'
    # # the ground truth
    # gold_path = data_path / 'gold.csv'
    # the analyzers used to convert the text into tokens for indexing
    analyzers = ['3gram']
    
    # initialize a local spark context
    spark = SparkSession.builder\
                        .master('local[*]')\
                        .appName('Sparkly Example')\
                        .getOrCreate()
    # read all the data as spark dataframes with tab as the separator
    table_a = spark.read.csv(index, header=True, inferSchema=True, sep=sep)
    table_a = table_a.withColumnRenamed(cid, "_id")
    table_b = spark.read.csv(query, header=True, inferSchema=True, sep=sep)
    table_b = table_b.withColumnRenamed(cid, "_id")
    cid = "_id"
    gold = spark.read.csv(gt, header=True, inferSchema=True, sep=sep)
    # the index config, '_id' column will be used as the unique 
    # id column in the index. Note id_col must be an integer (32 or 64 bit)
    config = IndexConfig(id_col=cid)
    # add the 'name' column to be indexed with analyzer above
    config.add_field(tid, analyzers)
    # create a new index stored at /tmp/example_index/
    index = LuceneIndex('/tmp/example_index/', config)
    # index the records from table A according to the config we created above
    index.upsert_docs(table_a)
    
    # get a query spec (template) which searches on 
    # all indexed fields
    query_spec = index.get_full_query_spec()
    # create a searcher for doing bulk search using our index
    searcher = Searcher(index)
    # search the index with table b
    candidates = searcher.search(table_b, query_spec, id_col=cid, limit=limit).cache()
    
    #candidates.show()
    # output is rolled up 
    # search record id -> (indexed ids + scores + search time)
    #
    # explode the results to compute recall
    pairs = candidates.select(
                        F.explode('ids').alias('a_id'),
                        F.col(cid).alias('b_id')
                    )
    # number of matches found
    true_positives = gold.intersect(pairs).count()
    # number of retrieved matches (candidates)
    retrieved_matches = pairs.count()
    
    # percentage of matches found
    recall = true_positives / gold.count()
    # precentage of matches found out of the retrieved matches (precision)
    precision = true_positives / retrieved_matches
    # F1 score (harmonic mean of precision and recall)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    candidates.unpersist()   
    return recall, precision, f1_score
    

# input_dir = '../../data/real/'
# log_file = '../../logs/logs_extended/sparkly.txt'
input_dir = '/usr/src/sparkly/data/'
log_file = '/usr/src/sparkly/logs/Sparkly.txt'

if __name__ == '__main__':
    
    cid, tid = 'id', 'aggregate value'
    stats = []
    with open(log_file, 'a') as out:
        for noc, (file1, file2, ground, sep, dir) in enumerate(cases):
            noc = f'D{noc+1}'
            gt = 'file://{}/{}/{}.csv'.format(input_dir, dir, ground)
            query = 'file://{}/{}/{}.csv'.format(input_dir, dir, file2)
            index = 'file://{}/{}/{}.csv'.format(input_dir, dir, file1)
            
            for k in [1, 5, 10]:
                t1 = time()
                recall, precision, f1 = run_sparkly(index=index, query=query,
                                                    gt=gt, sep=sep,
                                                    cid=cid, tid=tid, limit=k)
                t2 = time()
                
                print(dir, k, recall)
                out.write(json.dumps({'case': noc, 'k': k, 'time': t2-t1,
                                      'prec': precision, 'rec': recall,  'f1':f1})+"\n")
                out.flush()
                # break
            # break
