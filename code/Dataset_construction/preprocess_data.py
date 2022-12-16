from collections import defaultdict
import os
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

from sklearn import decomposition
from sklearn import preprocessing

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

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
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

if __name__ == "__main__":

    DATASET, pssmdir, radius, ngram, dim = sys.argv[1:]
    radius, ngram, dim = map(int, [radius, ngram, dim])

    with open(DATASET+'/dataset.txt', 'r') as f:
        data_list = f.read().strip().split('\n')

    feature_pd = pd.read_csv(DATASET+'/rdkit_final.csv')
    arrayT = feature_pd.values.reshape(feature_pd.shape[0], feature_pd.shape[1])


    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[1]]
    N = len(data_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    pssm_dict = defaultdict(lambda: len(pssm_dict))
    rdkitfeature_dict = defaultdict(lambda: len(rdkitfeature_dict))

    Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []

    pssms = []

    rdkitfeatures = []

    dir_input = (DATASET+'/input/'+
                 'radius' + str(radius) + '_ngram' + str(ngram) + '/')
    os.makedirs(dir_input, exist_ok=True)


    for no, data in enumerate(data_list):
        item, pname, smiles, sequence, interaction = data.strip().split()


        item = int(item)

        singlefeature = arrayT[item][1:]
        rdkitfeatures.append(singlefeature)

        singlepssm = load_pssm(pssmdir, pname)
        pssms.append(singlepssm)

        Smiles += smiles + '\n'

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        compounds.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        words = split_sequence(sequence, ngram)
        proteins.append(words)

        interactions.append(np.array([float(interaction)]))



    with open(dir_input + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'proteins', proteins)

    np.save(dir_input + 'pssms', pssms)

    np.save(dir_input + 'rdkitfeatures', rdkitfeatures)

    # interactions 的标准化处理
    # ss = preprocessing.StandardScaler()
    # interactions 的归一化处理
    # ss = preprocessing.MinMaxScaler()
    # std_interactions = ss.fit_transform(interactions)

    np.save(dir_input + 'interactions', interactions)
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_input + 'atom_dict.pickle')
    dump_dictionary(bond_dict, dir_input + 'bond_dict.pickle')
    dump_dictionary(edge_dict, dir_input + 'edge_dict.pickle')
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')

    print('The preprocess of ' + DATASET + ' dataset has finished!')
