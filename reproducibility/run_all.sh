#1. Setup of Experiments
## 1.1 Get Repo
#git clone https://github.com/alexZeakis/Embeddings4ER.git
#cd ../

## 1.2 Get Data
wget https://zenodo.org/record/8433873/files/data_ea.tar.gz
tar -xvzf data_ea.tar.gz
rm data_ea.tar.gz

## 1.3 Static Models
wget https://zenodo.org/records/11243756/files/models.tar.gz
tar -xvzf models.tar.gz
rm models.tar.gz

#2. Experiments
PREF="python/schema_agnostic/core/"

## 2.1 Vectorization
### 2.1.1 Vectorization on Real Data (*Exec 1a*)
python ${PREF}vectorize_real.py data/real/ embeddings/real/ logs/schema_agnostic/core/ models/
### 2.1.2 Vectorization on Synthetic Data (*Exec 1b*)
if [[ "$1" == "--synthetic" ]] || [[ "$2" == "--synthetic" ]]; then
  python ${PREF}vectorize_synthetic.py data/synthetic/profiles/ embeddings/synthetic/ logs/schema_agnostic/core/ models/
else
  cp or_logs/schema_agnostic/core/vectorization_synthetic.txt logs/schema_agnostic/core/
fi

## 2.2 Blocking
### 2.2.1 Blocking on Real Data (*Exec 2a*):
python ${PREF}blocking_real.py data/real/ embeddings/real/ logs/schema_agnostic/core/
### 2.2.2 Blocking on Synthetic Data (*Exec 2b*):
if [[ "$1" == "--synthetic" ]] || [[ "$2" == "--synthetic" ]]; then
  python ${PREF}blocking_synthetic.py data/synthetic/ground_truths/ embeddings/synthetic/ logs/schema_agnostic/core/
else
  cp or_logs/schema_agnostic/core/blocking_euclidean_synthetic.csv logs/schema_agnostic/core/
fi

## 2.3 Matching
### 2.3.1 Unsupervised Matching:
#### 2.3.1.1 Unsupervised Matching without blocking (*Exec 3a*):
python ${PREF}matching_unsupervised.py data/real/ embeddings/real/ logs/schema_agnostic/core/
#### 2.3.1.2 Unsupervised Matching with blocking (*Exec 3b*)
python ${PREF}matching_unsupervised_block.py data/real/ embeddings/real/ logs/schema_agnostic/core/

### 2.3.2 Supervised Matching
#### 2.3.2.1 Supervised Matching on static models (*Exec 4a*)
cd ${PREF}supervised/static/
python -m venv deepmatcher_venv

# Use a subshell to ensure the virtual environment is activated for the entire block
source deepmatcher_venv/bin/activate
pip install deepmatcher torch==1.9.0 torchtext==0.10
python transform_labeled.py ../../../../../data/labeled/ ../../../../../data/labeled_2/
chmod +x run_dm.sh
./run_dm.sh ../../../../../data/labeled_2/ ../../../../../logs/schema_agnostic/core/ ../../../../../models/
deactivate

# Cleanup
rm -r deepmatcher_venv
cd ../../../../../

#### 2.3.2.2 Supervised Matching on dynamic models (*Exec 4b*)
cd ${PREF}supervised/dynamic/
chmod +x matching_supervised.sh
./matching_supervised.sh ../../../../../data/labeled ../../../../../logs/schema_agnostic/core/ ../../../../../logs/schema_agnostic/core/sup_exps/
cd ../../../../../
            
## 2.4 Baseline
### 2.4.1 DeepBlocker
python python/baseline/DeepBlocker/run_DeepBlocker.py data/real/ logs/baseline/

### 2.4.2 ZeroER
# Check if the first argument is "--zeroer"
if [[ "$1" == "--zeroer" ]] || [[ "$2" == "--zeroer" ]]; then
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
