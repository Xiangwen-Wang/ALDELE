#!/usr/bin/python
# coding: utf-8

# Author: Xiangwen Wang
# Date: 2022-08-22

import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_squared_error,r2_score


class ConPrediction(nn.Module):
    def __init__(self, device, n_fingerprint, n_word, dim, layer_gnn, layer_cnn, window, layer_nn, layer_output):
        super(ConPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_nn = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_nn)])

        self.W_attention = nn.Linear(dim, dim)

        self.pairwise_transform = nn.Linear(dim, dim)
        self.p_out = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_nn)])

        self.W_out = nn.ModuleList([nn.Linear(3*dim, 3*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(3*dim, 1)

        self.device = device
        self.dim = dim
        self.layer_gnn = layer_gnn
        self.window = window
        self.layer_cnn = layer_cnn
        self.layer_nn = layer_nn
        self.layer_output = layer_output


    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.mean(xs, 0), 0)
        return xs

    def pssm_cnn(self, x, layer, dim):
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
        for i in range(layer):
            x = torch.relu(self.W_cnn[i](x))
        x = x.view(-1, dim)
        # x = torch.squeeze(torch.squeeze(x, 0), 0)
        h = torch.relu(self.W_attention(x))
        return torch.unsqueeze(torch.mean(h, 0), 0)

    def cnn(self, x, layer, dim):
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
        for i in range(layer):
            x = torch.relu(self.W_cnn[i](x))
        x = x.view(-1,dim)
        h = torch.relu(self.W_attention(x))
        return torch.unsqueeze(torch.mean(h, 0 ), 0)

    def nn(self, x, layer, dim):
        for i in range(layer):
            x = torch.relu(self.W_nn[i](x))
        x = x.view(-1, dim)
        h = torch.relu(self.W_attention(x))
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(h, 0), 0)

    def attention_p(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs
        attention_weights = F.linear(h,hs)[0].tolist()
        max_attention = max([float(attention) for attention in attention_weights])
        # print(max_attention)
        if max_attention == 0:
            attention_profiles_p = [0 for attention in attention_weights]
        else:
            attention_profiles_p = ['%.4f' %(float(attention)/max_attention) for attention in attention_weights]


        return torch.unsqueeze(torch.mean(ys, 0), 0), attention_profiles_p


    def attention_sub(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""


        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs
        attention_weights = F.linear(h,hs)[0].tolist()
        max_attention = max([float(attention) for attention in attention_weights])
        # print(max_attention)
        if max_attention == 0:
            attention_profiles_s = [0 for attention in attention_weights]
        else:
            attention_profiles_s = ['%.4f' %(float(attention)/max_attention) for attention in attention_weights]


        return torch.unsqueeze(torch.mean(ys, 0), 0), attention_profiles_s


    def pairwise_module(self, x, xs, layer, dim):
        pairwise_c_feature = F.leaky_relu(self.pairwise_transform(x))
        pairwise_p_feature = F.leaky_relu(self.pairwise_transform(xs))
        pairwise_pred = torch.matmul(pairwise_p_feature.transpose(0,1), pairwise_c_feature)
        hiden1 = torch.sigmoid(pairwise_pred)
        for i in range(layer):
            hiden1 = torch.relu(self.p_out[i](hiden1))
        hiden2 = hiden1.view(-1, dim)
        hiden2 = torch.relu(self.pairwise_transform(hiden2))
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(hiden2, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words, pssms, rdkitfeatures = inputs

        layer_cnn = 3
        layer_nn = 3
        layer_output = 3
        dim = 10
        layer_gnn = 3

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)
        """feature vector with NN."""
        rdkitfeature_vector = self.nn(rdkitfeatures, layer_nn, dim)
        substrate_vector, attention_profiles_s = self.attention_sub(rdkitfeature_vector,compound_vector, layer_cnn)

        """PSSM vector with pssm-CNN."""
        pssm_vector = self.pssm_cnn(pssms, layer_cnn, dim)
        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector, attention_profiles_p = self.attention_p(pssm_vector, word_vectors, layer_cnn)


        """pairwise vector"""
        pairwise_pred = self.pairwise_module(substrate_vector, protein_vector, layer_cnn, dim)

        """Concatenate the above vectors and output the interaction."""
        cat_vector = torch.cat((substrate_vector, protein_vector, pairwise_pred), 1)

        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction, attention_profiles_p, attention_profiles_s
    
    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction, weight_p, weight_s = self.forward(inputs)

        if train:
            loss = F.mse_loss(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            # correct_labels = correct_interaction.to('cpu').data.numpy()
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            # predicted_labels = list(map(lambda x: np.argmax(x), ys))
            # predicted_scores = list(map(lambda x: x[1], ys))
            return correct_values, predicted_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        testY, testPredict = [], []
        SAE = 0   # sum absolute error.
        for data in dataset:
            (correct_values, predicted_values) = self.model(data, train=False)
            SAE += sum(np.abs(predicted_values - correct_values))
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY, testPredict))
        r2 = r2_score(testY, testPredict)
        return MAE, rmse, r2

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle = True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_nn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_nn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_nn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = (DATASET+'/input/'
                 +'radius' + radius + '_ngram' + ngram + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.FloatTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    pssms = load_tensor(dir_input + 'pssms', torch.FloatTensor)
    rdkitfeatures = load_tensor(dir_input + 'rdkitfeatures', torch.FloatTensor)
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """Create a dataset and split it into train/dev/test."""

    dataset = list(zip(compounds, adjacencies, proteins, pssms, rdkitfeatures, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    torch.manual_seed(1234)
    model = ConPrediction(device, n_fingerprint, n_word, dim, layer_gnn, layer_cnn, window, layer_nn, layer_output).to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MAEs = DATASET+'/output_model5/results.txt'
    file_model = DATASET+'/output_model5/model'
    MAEs = ('Epoch\tTime(sec)\tLoss_train\tRMSE_train\tR2_train\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test')
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')
    print(file_MAEs)
    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        MAE_dev, RMSE_dev, R2_dev = tester.test(dataset_dev)
        MAE_test, RMSE_test, R2_test = tester.test(dataset_test)
        MAE_train, RMSE_train, R2_train = tester.test(dataset_train)
        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, loss_train, RMSE_train,
                R2_train, RMSE_dev, RMSE_test, R2_dev, R2_test]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
