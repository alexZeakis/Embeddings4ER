# Embedings4ER

## Overview

This repository contains the code, datasets and results of a thorough experimental analysis of 12 popular language models over 17 established benchmark datasets for the task of Entity Resolution. It is divided into three main categories: vectorization, blocking and matching. Our experimental results provide novel insights into the strengths and weaknesses of the main language models, facilitating researchers and practitioners to identify the most most suitable one in practical ER solutions.

## Models

 - Word2vec: [Gensim](https://radimrehurek.com/gensim/models/word2vec.html)
 - GloVe: [Hugging Face](https://huggingface.co/sentence-transformers/average_word_embeddings_glove.840B.300d)
 - FastText: [Gensim](https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText). For FastText please also download .bin file found [here]().
 - BERT: [Hugging Face](https://huggingface.co/bert-base-uncased) 
 - ALBERT: [Hugging Face](https://huggingface.co/albert-base-v2) 
 - RoBERTa: [Hugging Face](https://huggingface.co/roberta-base) 
 - DistilBERT: [Hugging Face](https://huggingface.co/distilbert-base-uncased) 
 - XLNet: [Hugging Face](https://huggingface.co/xlnet-base-cased)
 - S-MPNet: [Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 
 - S-T5: [Hugging Face](https://huggingface.co/sentence-transformers/gtr-t5-large) 
 - S-DistilRoBERTa: [Hugging Face](https://huggingface.co/sentence-transformers/all-distilroberta-v1) 
 - S-MINILM: [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)

## Experiment reproducibility

### Vectorization

- For vectorization on real datasets, please run:
```sh
python vectorize_real.py <output_dir>
```

- For vectorization on synthetic datasets, please run:
```sh
python vectorize_synthetic.py <output_dir>
```

### Blocking

- For blocking on real datasets, please run:
```sh
python blocking_real.py <input_dir>
```

- For blocking on synthetic datasets, please run:
```sh
python blocking_synthetic.py <input_dir>
```

### Matching

- For unsupervised matching, please run:
```sh
python matching_unsupervised.py <input_dir>
```

- For supervised matching, please run:
```sh
./matching_supervised.sh
```

### Datasets
Datasets are not currently in this repository, they can be downloaded externally [here]().

## Visualizations

To produce all plots, please run the jupyter notebook found [here](https://github.com/alexZeakis/Embedings4ER/blob/main/jupyter/Full_Plots.ipynb).

