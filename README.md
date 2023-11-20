# ALDELE: 

A deep-learning-based multiple toolkits (DeTool) approach that uses the inputs of enzymes and substrates for biocatalystic tasks.


## Introduction
This repository contains the PyTorch implementation of **ALDELE** framework.  
**ALDELE** is a deep learning framework with two-phase attention and pairwise module to explicitly learn non-covalent local interactions between enzymes and substrates for biocatlytic purpose.
It works on two-dimensional (2D) molecular graphs and physicochemical properties of compounds, and target protein sequences with amino acid evolutionary matrix to perform prediction.
## Framework
![1](https://github.com/Xiangwen-Wang/ALDELE/assets/83728171/6dbde8b3-4823-4f52-aaf6-ca2289edb716)
## System Requirements
The source code developed in Python 3.10. The required python dependencies are given below. 

```
numpy==1.23.1
packaging==21.3
pickleshare==0.7.5
py3Dmol==2.0.3
python-json-logger==2.0.7
rdkit==2023.3.2
scikit-learn==1.1.1
scipy==1.8.1
torch==1.12.0
torch-geometric==2.3.1
uri-template==1.3.0
urllib3==2.0.3
useful-rdkit-utils==0.2.7

```

## Datasets
The `datasets` folder contains all experimental data used in ALDELE.
 

## Data construction
the `code/Dataset_construction` folder contains the guidance of data preproceesing procedure.
Please check the Data_construction_protocol.doc for details of each step.


## Run ALDELE on Our Data to Reproduce Results

To train ALDELE, where we provide the basic configurations for all hyperparameters in `submit.sh`.
The folder name also need to modified based on the path you save your datasets.

```
$ bash submit.sh
```


## Acknowledgements
This implementation is inspired and partially based on earlier works [1], [2] and [3].


## References
    [1] Tsubaki, Masashi, Kentaro Tomii, and Jun Sese. "Compound–protein interaction prediction with end-to-end learning of neural networks for graphs and sequences." Bioinformatics 35.2 (2019): 309-318.
    [2] Li, Feiran, et al. "Deep learning-based k cat prediction enables improved enzyme-constrained model reconstruction." Nature Catalysis 5.8 (2022): 662-672.
    [3] Li, Shuya, et al. "MONN: a multi-objective neural network for predicting compound-protein interactions and affinities." Cell Systems 10.4 (2020): 308-322.
