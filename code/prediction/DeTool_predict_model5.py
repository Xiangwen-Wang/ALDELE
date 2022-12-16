#!/usr/bin/python
# coding: utf-8

# Author: Xiangwen Wang
# Date: 2022-08-22

import os
import math
import sys
import torch
import model5 as model
from rdkit import Chem
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

def load_pssm(pssmdir,name):
    all_points = []
    pssmline = []
    with open(pssmdir +'/'+ name + '.pssm',
              'r') as pssmfile:
        for line in pssmfile:
            point_tmp = line.split(' ')
            point_tmp = [x.strip() for x in point_tmp if x.strip() != ' ']
            point_tmp = list(map(float, point_tmp))
            all_points.append(point_tmp)

    return np.array(all_points)

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                try:
                    fingerprints.append(fingerprint_dict[fingerprint])
                except:
                    fingerprint_dict[fingerprint] = 0
                    fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    try :
                        edge = edge_dict[(both_side, edge)]
                    except :
                        edge_dict[(both_side, edge)] = 0
                        edge = edge_dict[(both_side, edge)]

                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    # words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]

    words = list()
    for i in range(len(sequence)-ngram+1):
        try:
            words.append(word_dict[sequence[i:i + ngram]])
        except:
            word_dict[sequence[i:i + ngram]] = 0
            words.append(word_dict[sequence[i:i + ngram]])

    return np.array(words)
    # return word_dict

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        predicted_value = self.model.forward(data)

        return predicted_value

if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, pssmdir, radius, ngram, dim, window, layer_gnn, layer_cnn, layer_nn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]

    (dim, window, layer_gnn, layer_cnn, layer_nn, layer_output, decay_interval,
     iteration) = map(int, [dim, window, layer_gnn, layer_cnn, layer_nn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    dir_input = (DATASET+'/input/'
                 +'radius' + radius + '_ngram' + ngram + '/')
    ngram = int(ngram)

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    word_dict = model.load_pickle(dir_input+'word_dict.pickle')
    n_word = len(word_dict)
    fingerprint_dict = model.load_pickle(dir_input+'fingerprint_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    atom_dict = model.load_pickle(dir_input+'atom_dict.pickle')
    bond_dict = model.load_pickle(dir_input+'bond_dict.pickle')
    edge_dict = model.load_pickle(dir_input+'edge_dict.pickle')

    # torch.manual_seed(1234)
    predict_model = model.ConPrediction(device, n_fingerprint, n_word, dim, layer_gnn, layer_cnn, window, layer_nn,
                                                        layer_output).to(device)
    predict_model.load_state_dict(torch.load(DATASET+'/output_model5/model',
        map_location=device))



    predictor = Predictor(predict_model)

    with open(DATASET+'/output_model5/prediction/dataset.txt', 'r') as f:
        data_list = f.read().strip().split('\n')

    feature_pd = pd.read_csv(DATASET+'/output_model5/prediction/rdkit_final.csv')
    arrayT = feature_pd.values.reshape(feature_pd.shape[0], feature_pd.shape[1])

    print(len(data_list))  # 6291
    print('\n')

    # csv file with protein sequences and pssm files name (finish from PSI-BLAST, saved in the pssm_dir)
    first_line = ('pname\tSMILES\tsequence\tPrediction')
    with open(DATASET+'/output_model5/prediction/prediction_results.txt', 'w') as file:
        file.write(first_line + '\n')
        predict_values = list()
        for no, line in enumerate(data_list):
            item, pname, smiles, sequence, interaction = line.strip().split()

            singlepssm = load_pssm(pssmdir, pname)
            pssms = torch.FloatTensor(singlepssm)

            words = split_sequence(sequence, ngram)
            words = torch.LongTensor(words)

            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
            atoms = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)
            radius = int(radius)
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
            adjacency = create_adjacency(mol)
            fingerprints = torch.LongTensor(fingerprints)
            adjacency = torch.FloatTensor(adjacency)

            item = int(item)
            singlefeature = arrayT[item][1:]
            rdkitfeatures = torch.FloatTensor(singlefeature)

            inputs = [fingerprints, adjacency, words, pssms, rdkitfeatures]
            prediction, attention_p, attention_s = predictor.predict(inputs)
            predict_value = prediction.item()
            print('%.4f' % (predict_value))
            predict_values.append(predict_value)
            predict = [pname, smiles, sequence, predict_value]
            file.write('\t'.join(map(str, predict)) + '\n')


