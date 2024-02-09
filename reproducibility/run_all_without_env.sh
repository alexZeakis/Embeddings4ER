#1. Setup of Experiments
## 1.1 Get Repo
#git clone https://github.com/alexZeakis/Embeddings4ER.git
cd ../

## 1.2 Get Data
wget https://zenodo.org/record/8433873/files/data_ea.tar.gz
tar -xvzf data_ea.tar.gz
rm data_ea.tar.gz

## 1.3 Static Models
mkdir models
cd models/

### 1.3.1 FastText
mkdir fasttext
cd fasttext/
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip
rm wiki.en.zip
cd ..

### 1.3.2 Word2Vec
mkdir word2vec
cd word2vec/
pip install gdown
#gdown https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
python ../../reproducibility/dl_word2vec.py
apt-get install gunzip
gunzip GoogleNews-vectors-negative300.bin.gz
cd ../..

#2. Experiments
PREF="python/schema_agnostic/core/"
#Remove existing logs to create new ones
#rm -r logs/
mv logs/ or_logs/  # Change "rm -r logs/" to "mv logs/ or_logs/"

## 2.1 Vectorization
### 2.1.1 Vectorization on Real Data (*Exec 1a*)
python "${PREF}vectorize_real.py" data/real/ embeddings/real/ logs/schema_agnostic/core/ models/
### 2.1.2 Vectorization on Synthetic Data (*Exec 1b*)
python "${PREF}vectorize_synthetic.py" data/synthetic/profiles/ embeddings/synthetic/ logs/schema_agnostic/core/ models/

## 2.2 Blocking
### 2.2.1 Blocking on Real Data (*Exec 2a*):
python "${PREF}blocking_real.py" data/real/ embeddings/real/ logs/schema_agnostic/core/
### 2.2.2 Blocking on Synthetic Data (*Exec 2b*):
python "${PREF}blocking_synthetic.py" data/synthetic/ground_truths/ embeddings/synthetic/ logs/schema_agnostic/core/


## 2.3 Matching
### 2.3.1 Unsupervised Matching:
#### 2.3.1.1 Unsupervised Matching without blocking (*Exec 3a*):
python "${PREF}matching_unsupervised.py" data/real/ embeddings/real/ logs/schema_agnostic/core/
#### 2.3.1.2 Unsupervised Matching with blocking (*Exec 3b*)
python "${PREF}matching_unsupervised_block.py" data/real/ embeddings/real/ logs/schema_agnostic/core/

### 2.3.2 Supervised Matching
#### 2.3.2.1 Supervised Matching on static models (*Exec 4a*)
cd python/schema_agnostic/core/supervised/static/
python transform_labeled.py ../../../../../data/labeled/ ../../../../../data/labeled_2/
./run_dm.sh ../../../../../data/labeled_2/ ../../../../../logs/schema_agnostic/core/ ../../../../../models/
cd ../../../../../

#### 2.3.2.2 Supervised Matching on dynamic models (*Exec 4b*)
cd python/schema_agnostic/core/supervised/dynamic/
./matching_supervised.sh ../../../../../data/labeled ../../../../../logs/schema_agnostic/core/ ../../../../../logs/schema_agnostic/core/sup_exps/
cd ../../../../../
            
## 2.4 Baseline
### 2.4.1 DeepBlocker
python python/baseline/DeepBlocker/run_DeepBlocker.py data/real/ logs/baseline/

### 2.4.2 ZeroER
# Check if the first argument is "--zeroer"
if [[ "$1" == "--zeroer" ]]; then
  # 2.4.2 ZeroER
  cd python/baseline/ZeroER/
  conda env create -f environment.yml
  conda deactivate
  conda activate ZeroER
  ./create_data.sh
  ./run.sh ../../../logs/baseline/
  cd ../../../
else
  # Copy the ZeroER.txt file from or_logs/baseline/ to logs/baseline/
  cp or_logs/baseline/ZeroER.txt logs/baseline/
fi

# 3 Visualizations
rsync -av --ignore-existing or_logs/baseline/ logs/baseline/
rsync -av --ignore-existing or_logs/schema_agnostic/core/ logs/schema_agnostic/core/

cd visualizations/
python Schema-Agnostic-Core.py
conda deactivate

# 4 Cleanup
#cd ../
#rm -r Embeddings4ER/
#conda remove --name vldb23_ea_basic --all
#conda remove --name vldb23_ea_basicV4 --all  
#conda remove --name ZeroER --all 
