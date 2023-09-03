#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''
#########################################################################################################

   Version:  21 May 2022

   Author:   Xiangwen Wang (x.wang@qub.ac.uk)

   Purpose:  Convert residues name weighted energy file into 2D arrays.

   Parameters:
       Residues name:
            ALA - 1
            ARG - 2
            ASN - 3
            ASP - 4
            CYS - 5
            GLU - 6
            GLN - 7
            GLY - 8
            HIS - 9
            ILE - 10
            LEU - 11
            LYS - 12
            MET - 13
            PHE - 14
            PRO - 15
            SER - 16
            THR - 17
            TRP - 18
            TYR - 19
            VAL - 20

       Weighted energy file: https://www.rosettacommons.org/docs/latest/rosetta_basics/scoring/score-types

        Output：
            2D matrix:  (N, 1) + (N, F)
            N is the number of residues, F is the number of features of weighted energy file.
    unpublished codes

###########################################################################################################
'''
import os
import numpy as np
import fileinput
import codecs

def single_file(fi, output):
    amino_acid_mapping = {
        "ALA": 1,
        "ARG": 2,
        "ASN": 3,
        "ASP": 4,
        "CYS": 5,
        "GLU": 6,
        "GLN": 7,
        "GLY": 8,
        "HIS": 9,
        "ILE": 10,
        "LEU": 11,
        "LYS": 12,
        "MET": 13,
        "PHE": 14,
        "PRO": 15,
        "SER": 16,
        "THR": 17,
        "TRP": 18,
        "TYR": 19,
        "VAL": 20
    }

    anumber = 21

    with open(output, 'w') as outputfile:
        for line, strin in enumerate(fileinput.input(fi,encoding='latin1',errors='ignore')):
            if line >= 4:
                aa = strin.split()[0][0:3]
                str_vec = strin.split()[1:]
                if aa in amino_acid_mapping:
                    anumber = amino_acid_mapping[aa]
                outputfile.write(f"{anumber} {' '.join(str_vec)}\n")



    outputfile.close()
    fileinput.close()

if __name__ == '__main__':
    dirpath = 'yourpath of weighted energy files'
    orfolder = dirpath + 'original/'
    outputfolder = dirpath + 'output/'
    listfile = os.listdir(orfolder)
    for eachfile in listfile:
        fi = orfolder + eachfile  # input
        output = outputfolder + '/' + eachfile
        print(eachfile)
        print("\n")
        single_file(fi, output)
    print("\n\ntask finish. Check the file path.")