import pandas as pd
import torch

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


def cosine_similarity(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

