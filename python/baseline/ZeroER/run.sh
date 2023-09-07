#!/bin/bash

log=$1

python zeroer.py rest1_rest2 D1 "$log" --run_transitivity --LR_dup_free --sep "|"
python zeroer.py abt_buy D2 "$log" --run_transitivity --LR_dup_free --sep "|"
python zeroer.py amazon_gp D3 "$log" --run_transitivity --LR_dup_free --sep "#"
python zeroer.py dblp_acm D4 "$log" --run_transitivity --LR_dup_free --sep "%"
python zeroer.py imdb_tvdb D5 "$log" --run_transitivity --LR_dup_free --sep "|"
python zeroer.py tmdb_tvdb D6 "$log" --run_transitivity --LR_dup_free --sep "|"
python zeroer.py imdb_tmdb D7 "$log" --run_transitivity --LR_dup_free --sep "|"
python zeroer.py walmart_amazon D8 "$log" --run_transitivity --LR_dup_free --sep "|"
python zeroer.py dblp_scholar D9 "$log" --run_transitivity --LR_dup_free --sep ">"
python zeroer.py imdb_dbpedia D10 "$log" --run_transitivity --LR_dup_free --sep "|"
