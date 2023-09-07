'''
cases = [('rest1', 'rest2', 'gt', "|", 'rest1_rest2', ['name', 'phone_number', 'aggregate value']),
        ('abt', 'buy', 'gt', "|", 'abt_buy', ['name', 'description', 'aggregate value']),
        ('amazon', 'gp', 'gt', "#", 'amazon_gp', ['title', 'description', 'aggregate value']),
        ('dblp', 'acm', 'gt', "%", 'dblp_acm', ['title', 'authors', 'aggregate value']),
        ('imdb', 'tvdb', 'gtImTv', "|", 'imdb_tvdb', ['title', 'name', 'aggregate value']),
        ('tmdb', 'tvdb', 'gtTmTv', "|", 'tmdb_tvdb',  ['title', 'name', 'aggregate value']),
        ('imdb', 'tmdb', 'gtImTm', "|", 'imdb_tmdb',  ['title', 'name', 'aggregate value']),
        ('dblp', 'scholar', 'gt', ">", 'dblp_scholar',  ['title', 'authors', 'aggregate value']),
        ('imdb', 'dbpedia', 'gtImDb', "|", 'imdb_dbpedia',  ['title', 'starring', 'aggregate value']),
        ]

cases = [('walmart', 'amazon', 'gt', "|", 'walmart_amazon', ['title', 'modelno', 'aggregate value', 'aggregate value'])]
'''

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
