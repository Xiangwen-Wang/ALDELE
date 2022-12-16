#!/usr/bin/python
# coding: utf-8

# Author: Xiangwen Wang
# Date: 2022-08-22

import os
import math
import sys
import torch
import run_training_3_4
import json
import pickle
import numpy as np
from collections import defaultdict
from scipy import stats

def load_pssm(pssmdir,name):
    all_points = []
    with open(pssmdir +'/'+ name + '.pssm',
              'r') as pssmfile:
        for line in pssmfile:
            point_tmp = line.split(' ')
            point_tmp = [x.strip() for x in point_tmp if x.strip() != ' ']
            point_tmp = list(map(float, point_tmp))
            all_points.append(point_tmp)
    return np.array(all_points)

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

        return predicted_value, attention_profiles

if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, pssmdir, modelname, radius, ngram, dim, window, layer_cnn, layer_nn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration, setting) = sys.argv[1:]

    (dim, window, layer_cnn, layer_nn, layer_output, decay_interval,
     iteration) = map(int, [dim, window, layer_cnn, layer_nn, layer_output,
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

    word_dict = run_training_3_4.load_pickle(dir_input+'word_dict.pickle')
    n_word = len(word_dict)

    # torch.manual_seed(1234)
    Kcat_model = run_training_3_4.ConPrediction(device, n_word, dim, layer_cnn, window, layer_nn,
                                                        layer_output).to(device)
    Kcat_model.load_state_dict(torch.load(DATASET+'/'+modelname+'/model',
        map_location=device))



    predictor = Predictor(Kcat_model)

    with open(DATASET+'/'+modelname+'/prediction/dataset.txt', 'r') as f:
        data_list = f.read().strip().split('\n')


    print(len(data_list))  # 6291
    print('\n')

    # csv file with protein sequences and pssm files name (finish from PSI-BLAST, saved in the pssm_dir)
    first_line = ('pname\tsequence\tTm')
    with open(DATASET+'/'+modelname+'/prediction/prediction_results.txt', 'w') as file:
        file.write(first_line + '\n')

        Kcat_values = list()

        for no, line in enumerate(data_list):
            pname, sequence, interactions = line.strip().split()

            words = split_sequence(sequence, ngram)
            words = torch.LongTensor(words)

            singlepssm = load_pssm(pssmdir, pname)
            pssms = torch.FloatTensor(singlepssm)

            inputs = [words, pssms]
            prediction = predictor.predict(inputs)
            Kcat_value = prediction.item()
            print('%.4f' % (Kcat_value))
            Kcat_values.append(Kcat_value)
            KCAT = [pname, sequence, Kcat_value]
            file.write('\t'.join(map(str, KCAT)) + '\n')


