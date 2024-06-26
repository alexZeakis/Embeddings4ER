import pandas as pd

vectorizers = ['word2vec', 'fasttext', 'glove',
               'bert', 'distilbert', 'roberta', 'xlnet', 'albert', 
               'smpnet', 'st5', 'sdistilroberta', 'sminilm']


cases = [
        ('rest1', 'rest2', 'gt', "|", 'D1(rest)',
             [('http:www.okkam.orgontology_restaurant1.owl#name', 'http:www.okkam.orgontology_restaurant2.owl#name'),
              ('http:www.okkam.orgontology_restaurant1.owl#phone_number', 'http:www.okkam.orgontology_restaurant2.owl#phone_number'),
              ('aggregate value', 'aggregate value')]),
        ('abt', 'buy', 'gt', "|", 'D2(abt-buy)',
             [('name', 'name'), ('description', 'description'), ('aggregate value', 'aggregate value')]),
        ('amazon', 'gp', 'gt', "#", 'D3(amazon-gp)',
            [('title', 'title'), ('description', 'description'), ('aggregate value', 'aggregate value')]),
        ('dblp', 'acm', 'gt', "%", 'D4(dblp-acm)', 
            [('title', 'title'), ('authors', 'authors'), ('aggregate value', 'aggregate value')]),
        ('imdb', 'tvdb', 'gtImTv', "|", 'D5_D6_D7(imdb-tmdb)',
             [('https:www.scads.demovieBenchmarkontologytitle', 'https:www.scads.demovieBenchmarkontologytitle'),
                 ('https:www.scads.demovieBenchmarkontologyname', 'https:www.scads.demovieBenchmarkontologyname'),
                 ('aggregate value', 'aggregate value')]),
        ('tmdb', 'tvdb', 'gtTmTv', "|", 'D5_D6_D7(imdb-tmdb)', 
            [('https:www.scads.demovieBenchmarkontologytitle', 'https:www.scads.demovieBenchmarkontologytitle'),
                 ('https:www.scads.demovieBenchmarkontologyname', 'https:www.scads.demovieBenchmarkontologyname'),
                 ('aggregate value', 'aggregate value')]),
        ('imdb', 'tmdb', 'gtImTm', "|", 'D5_D6_D7(imdb-tmdb)', 
            [('https:www.scads.demovieBenchmarkontologytitle', 'https:www.scads.demovieBenchmarkontologytitle'),
                 ('https:www.scads.demovieBenchmarkontologyname', 'https:www.scads.demovieBenchmarkontologyname'),
                 ('aggregate value', 'aggregate value')]),
        ('walmart', 'amazon', 'gt', "|", 'D8(walmart-amazon)', 
             [('title', 'title'), ('modelno', 'modelno'), ('aggregate value', 'aggregate value')]),
        ('dblp', 'scholar', 'gt', ">", 'D9(dblp-scholar)', 
             [('title', 'title'), ('authors', 'authors'), ('aggregate value', 'aggregate value')]),
        ('imdb', 'dbpedia', 'gtImDb', "|", 'D10(movies)', 
             [('title', 'title'), ('starring', 'actor name'), ('aggregate value', 'aggregate value')]),
        ]
pd.DataFrame(cases)
