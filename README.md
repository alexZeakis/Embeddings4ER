# Embedings4ER

## Overview

This repository contains the code, datasets and results of a thorough experimental analysis of 12 popular language models over 17 established benchmark datasets for the task of Entity Resolution. It is divided into three main categories: vectorization, blocking and matching. Our experimental results provide novel insights into the strengths and weaknesses of the main language models, facilitating researchers and practitioners to identify the most most suitable one in practical ER solutions.

## Models

 - Word2vec: [Gensim](https://radimrehurek.com/gensim/models/word2vec.html)
 - GloVe: [Hugging Face](https://huggingface.co/sentence-transformers/average_word_embeddings_glove.840B.300d)
 - FastText: [Gensim](https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText). For FastText please also download .bin file found [here](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip).
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

### Schema-Agnostic - Core

For the core experiments run on schema-agnostic settings and referenced in the original paper,
please visit [this link](https://github.com/alexZeakis/Embeddings4ER/tree/main/python/schema_agnostic/core/).

### Schema-Agnostic - Extended
For the extended experiments run on schema-agnostic settings,
please visit [this link](https://github.com/alexZeakis/Embeddings4ER/tree/main/python/schema_agnostic/extended/).

### Schema-Based - Core
For the core experiments run on schema-based settings,
please visit [this link](https://github.com/alexZeakis/Embeddings4ER/tree/main/python/schema_based/core/).

### Baseline
For the baseline experiments run on schema-agnostic settings,
please visit [this link](https://github.com/alexZeakis/Embeddings4ER/tree/main/python/baseline/).




### Datasets
Datasets are not currently in this repository, they can be downloaded externally [here](https://zenodo.org/record/8433873/files/data_ea.tar.gz).

### Static Models
For static models, please create a local directory with any given name, but inside create two directories:
 - One called `fasttext/` and inside download the file found [here](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip).
 - One called `word2vec/` and inside download the file found [here](https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download). To download a GDrive file, one can use the tool [GDown](https://github.com/wkentaro/gdown), which is used like `wget` but for GDrive links. It can be installed via PyPI with `pip install gdown`.

This outer directory will be used in various executions. More instructions on each page.

## Visualizations

To produce all plots, please run the jupyter notebook found [here](https://github.com/alexZeakis/Embedings4ER/blob/main/jupyter/Full_Plots.ipynb).

