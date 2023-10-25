cp -r ../../../data/real datasets
cd datasets/
mv "D1(rest)" rest1_rest2
echo -e "rest1.csv\nrest2.csv\ngt.csv" >> rest1_rest2/metadata.txt
mv "D2(abt-buy)" abt_buy 
echo -e "abt.csv\nbuy.csv\ngt.csv" >> abt_buy/metadata.txt
mv "D3(amazon-gp)" amazon_gp
echo -e "amazon.csv\ngp.csv\ngt.csv" >> amazon_gp/metadata.txt
mv "D4(dblp-acm)" dblp_acm
echo -e "dblp.csv\nacm.csv\ngt.csv" >> dblp_acm/metadata.txt
mv "D5_D6_D7(imdb-tmdb)" imdb_tvdb
echo -e "imdb.csv\ntvdb.csv\ngtImTv.csv" >> imdb_tvdb/metadata.txt
cp -r imdb_tvdb tmdb_tvdb
echo -e "tmdb.csv\ntvdb.csv\ngtTmTv.csv" >> tmdb_tvdb/metadata.txt
cp -r imdb_tvdb imdb_tmdb
echo -e "imdb.csv\ntmdb.csv\ngtImTm.csv" >> imdb_tmdb/metadata.txt
mv "D8(walmart-amazon)" walmart_amazon
echo -e "walmart.csv\namazon.csv\ngt.csv" >> walmart_amazon/metadata.txt
mv "D9(dblp-scholar)" dblp_scholar
echo -e "dblp.csv\nscholar.csv\ngt.csv" >> dblp_scholar/metadata.txt
mv "D10(movies)" imdb_dbpedia
echo -e "imdb.csv\ndbpedia.csv\ngtImDb.csv" >> imdb_dbpedia/metadata.txt