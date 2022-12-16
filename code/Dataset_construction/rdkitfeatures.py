import os
import rdkit
import rdkit.Chem
from rdkit.Chem import AllChem
from rdkit import Chem
import rdkit.Chem.Crippen as Crippen
import pandas as pd
import numpy as np
import csv
import useful_rdkit_utils
from useful_rdkit_utils import RDKitDescriptors
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def rdkit_feature(interaction,smiles):
    columnlist = []
    columnlist.append(interaction)
    desc_calc = RDKitDescriptors()
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for n, v in zip(desc_calc.desc_names, desc_calc.calc_mol(mol)):
        columnlist.append(v)
    return columnlist

def rdkit_all(inputdir):
    with open(inputdir+'pre_dataset.txt', 'r') as f:
        data_list = f.read().strip().split('\n')
    data_list = [d for d in data_list if '.' not in d.strip().split()[1]]
    desc_calc = RDKitDescriptors()
    columnlist =[]
    with open(inputdir + 'rdkit_all.csv', 'w') as outputfile:
        with open(inputdir + 'dataset.txt','w') as MLinputfile:
            writer = csv.writer(outputfile)
            columnlist.append('interaction')
            for n in desc_calc.desc_names:
                columnlist.append(n)
            writer.writerow(columnlist)
            for no, data in enumerate(data_list):
                pname, smiles, sequence, interaction = data.strip().split()
                interaction = float(interaction)
                MLinputfile.write(f'{no}\t{pname}\t{smiles}\t{sequence}\t{interaction}\n')
                columnlist = rdkit_feature(interaction,smiles)
                writer.writerow(columnlist)
    outputfile.close()
    MLinputfile.close()

def RF_ranking(inputdir):
    df = pd.read_csv(inputdir + 'rdkit_all.csv',encoding='utf-8')
    df = df.fillna(0)
    df.columns = list(df)
    X = df.iloc[:,1:]
    X1 = np.nan_to_num(X.astype(np.float32))
    y = df.iloc[:,0]
    feat_labels = df.columns[1:]
    X_train, X_test, y_train, y_test = train_test_split(X1,y,test_size=0.2, random_state =0)

    # PCA
    # estimators = [('linear_pca',PCA()),('kernel_pca',KernelPCA())]
    # pca = decomposition.PCA(n_components = dim)
    # X_features = pca.fit(X).transform(X)

    # RF 
    RF = RandomForestRegressor()
    RF.fit(X_train,y_train)
    importance = RF.feature_importances_
    indices = np.argsort(importance)[::-1]
    remainlist = []
    with open(inputdir+'RFimportance.txt','w') as importancefile:
        for f in range(10):
            importancefile.write("%2d) %-*s %f" % \
                                 (f+1, 30,feat_labels[indices[f]],importance[indices[f]]))
            importancefile.write('\n')
            remainlist.append(feat_labels[indices[f]])
    importancefile.close()
    x_selected = X[remainlist]
    x_selected_columns = remainlist
    x_select_pd = pd.DataFrame(x_selected, columns =x_selected_columns)

    pd.DataFrame(x_select_pd).to_csv(inputdir+'rdkit_final.csv')


if __name__ == "__main__":

    inputdir = './'

    rdkit_all(inputdir)
    print('finish rdkit_all.csv\n')

    RF_ranking(inputdir)
    print('finish rdkit_features ranking and selection\n')


