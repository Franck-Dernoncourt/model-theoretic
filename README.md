Data, code, and result log for TODO.

To run the code:
```shell
cd code

python xp.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-onlyspaces.txt > ad_word2vec_plsr.txt & 
python xp.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-framenet.txt > ad_word2vec-framenet_plsr.txt & 
python xp.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-ppdb.txt > ad_word2vec-ppdb_plsr.txt & 
python xp.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-wordnet-synonyms.txt > ad_word2vec-wordnet_plsr.txt & 
python xp.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-wordnet-synonyms-plus.txt > ad_word2vec-wordnet-plus_plsr.txt & 
python xp.py -m PLSR -s ad -v random > ad_random_plsr.txt & 
    
python xp.py -m nn -s ad -v ../data/GoogleNews-vectors-negative300-short-onlyspaces.txt > ad_word2vec_nn.txt & 
python xp.py -m nn -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-framenet.txt > ad_word2vec-framenet_nn.txt & 
python xp.py -m nn -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-ppdb.txt > ad_word2vec-ppdb_nn.txt & 
python xp.py -m nn -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-wordnet-synonyms.txt > ad_word2vec-wordnet_nn.txt & 
python xp.py -m nn -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-wordnet-synonyms-plus.txt > ad_word2vec-wordnet-plus_nn.txt & 
python xp.py -m nn -s ad -v random > ad_random_nn.txt & 
    
python xp.py -m mode -s ad  > ad_mode.txt & 
python xp.py -m true-mode -s ad > ad_true-mode.txt & 
```
