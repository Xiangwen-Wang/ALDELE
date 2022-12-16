#!/usr/bin/python
# coding: utf-8

import math
import json
import pickle
import numpy as np
from collections import defaultdict
from rdkit import Chem

proteins = []


def fasta_save(sequence, num):
    dirpath = './fasta/'
    with open(dirpath+str(num)+'.fasta','w') as pssmfile:
        pssmfile.write(f'>num{num} \n')
        pssmfile.write(sequence)


def main() :
    inputdir = 'DATASET_dir_path'
    with open(inputdir+'raw_data.txt', 'r') as f:
        data_list = f.read().strip().split('\n')
    i = 0
    j = 0
    proteins = []
    with open(inputdir + 'pre_dataset.txt', 'w') as outputfile:
        for no, data in enumerate(data_list):
            item, sequence, smiles, interaction = data.strip().split()
            interaction = float(interaction)
            if  i == 0:
                outputfile.write(f'{i}\t{smiles}\t{sequence}\t{interaction}\n')
                proteins.append(sequence)
                fasta_save(sequence, i)
                i = i + 1
            else:
                if sequence in proteins:
                    for l in range(i):
                        if sequence == proteins[l]:
                            outputfile.write(f'{l}\t{smiles}\t{sequence}\t{interaction}\n')
                            break
                else:
                    outputfile.write(f'{i}\t{smiles}\t{sequence}\t{interaction}\n')
                    proteins.append(sequence)
                    fasta_save(sequence, i)
                    i = i + 1
            print(f'pssm number {i} \n')

if __name__ == '__main__' :
    main()