#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''
#########################################################################################################

   Version:  21 May 2022

   Author:   Xiangwen Wang (x.wang@qub.ac.uk)

   Purpose:  Compute the smoothed PSSM with given sliding window size and
   Shannon entropy from PSI-BLAST

   Input:    dir of the input files and output files, sliding window size

    unpublished codes

###########################################################################################################
'''
import os
import numpy as np
import fileinput
import codecs

def single_file(fi, output_smth, w_smth, output_PSSM, output_wop, outfile_lfc, outfile_Ent, output_full):
    w_smth = int(w_smth)

    PSSM = []
    WOP = []
    seq_cn = 0
    # PSI-BLAST file reading and separate to PSSM and WOP
    with open(output_PSSM, 'w') as pssmorigin, open(output_wop, 'w') as woporigin:
        for line, strin in enumerate(fileinput.input(fi,encoding='latin1',errors='ignore')):
            if line <= 2:
                continue
            if not len(strin.strip()):
                break

            str_vec = strin.split()[1:22]
            PSSM.append(str_vec[1:])

            str_vec_wop = strin.split()[22:-2]
            WOP.append(str_vec_wop)

            seq_cn += 1
            single_line_save(strin, pssmorigin, woporigin)
    print(seq_cn)
    pssmorigin.close()
    woporigin.close()
    fileinput.close()

    n = int(seq_cn/((w_smth-1)/2))
    every = int((w_smth-1)/2)
    n=n-1

    # EntWOP
    WOP_origst = np.array(WOP)
    WOP_orig = []
    WOP_orig = np.array([[0.0] * 20] * seq_cn)
    print(WOP_origst)
    for j in range(seq_cn):
        for i in range(20):
            WOP_orig[j,i] = float(WOP_origst[j,i])
    Ent_WOP = WOP_Ent(WOP_orig,n,every)
    with open(outfile_Ent,'w') as Entorigin:
        for i in range(n):
            Entorigin.write("%.2f"%Ent_WOP[i]+'\t')
        # np.savetxt(Entorigin, Ent_WOP, fmt="%.2f")
    Entorigin.close()
    # PSSM standard logistic function calculation f(x)=1/(1+exp(-x))
    # PSSM change format
    PSSM_origst = np.array(PSSM)
    PSSM_smth = np.array([[0.0] * 20] * seq_cn)
    PSSM_orig = np.array([[0.0] * 20] * seq_cn)
    for j in range(seq_cn):
        for i in range(20):
            PSSM_orig[j,i] = 1/(1+np.exp(-int(PSSM_origst[j,i])))
            # PSSM_orig[j,i] = float(PSSM_origst[j,i])

    # write full sequence standard logistic function to lfc folder
    with open(outfile_lfc,'w') as lfcorigin:
        np.savetxt(lfcorigin, PSSM_orig, fmt="%.2f")

    # full sequence sliding window results
    PSSM_smth_full = pssm_smth(PSSM_orig, PSSM_smth, w_smth, seq_cn)

    with open(output_full,'w') as outfile:
        np.savetxt(outfile, PSSM_smth_full, fmt="%.2f")

    # select the sliding window size n of full smoothed results
    PSSM_smth_final = PSSM_sliding(PSSM_smth_full,n,every)

    # write sliding window results as output
    with open(output_smth,'w') as outfile:
        np.savetxt(outfile, PSSM_smth_final, fmt="%.2f")

def single_line_save(strin, pssmorigin, woporigin):
    # write pssm origin and wop origin
    col = strin[0:5].strip()
    col += '\t' + strin[5:8].strip()
    wop = col
    begin = 10
    end = begin + 3
    for i in range(20):
        end = begin + 3
        col += '\t' + strin[begin:end].strip()
        begin = end + 1
    begin = 91
    for i in range(20):
        end = begin + 3
        wop += '\t' + strin[begin:end].strip()
        begin = end + 1
    col += '\n'
    wop += '\n'
    pssmorigin.write(''.join(col))
    woporigin.write(''.join(wop))

def WOP_Ent(WOP_orig, n, every):
    Ent_WOP = []
    for array in WOP_orig:
        p_sum = np.sum(array)
        p = array / float(p_sum)
        se = 0
        for p_i in p:
            se += p_i * np.log(p_i+0.00000001)
        se = abs(se)
        Ent_WOP.append(se)
    linenum = 0
    Ent_WOP_final = []
    for i in range(n):
        linenum += every
        Ent_WOP_final.append(Ent_WOP[linenum])
    return Ent_WOP_final

def PSSM_sliding(PSSM_smth_full,n,every):
    PSSM_smth_final = [[0.0] * 20] * n
    linenum = 0
    for i in range(n):
        linenum += every
        PSSM_smth_final[i] = PSSM_smth_full[linenum]
    return PSSM_smth_final

def pssm_smth(PSSM_orig, PSSM_smth, w_smth, l):
    for i in range(l):
        # smooth sliding window beyond the pssm top border
        if i < (w_smth - 1) / 2:
            for j in range(int(i + (w_smth - 1) / 2 + 1)):
                PSSM_smth[i] += PSSM_orig[j]
        # smooth sliding window beyond the pssm bottom border
        elif i >= (l - (w_smth - 1) / 2):
            for j in range(int(i - (w_smth - 1) / 2), l):
                PSSM_smth[i] += PSSM_orig[j]
        else:
            for j in range(int(i - (w_smth - 1) / 2), int(i + (w_smth - 1) / 2 + 1)):
                PSSM_smth[i] += PSSM_orig[j]
    return PSSM_smth

if __name__ == '__main__':
    dirpath = '/Users/xw0201/Desktop/postdoc/paper_dataset/Phosphatase/pssm/'
    pssmdir = dirpath + 'originresults'
    newdir = dirpath + 'standard'
    wopdir = dirpath + 'wop'
    swdir = dirpath + 'slidingwindow'
    fulldir = dirpath + 'swfull'
    lfcdir = dirpath + 'lfc'
    Entdir = dirpath + 'EntWOP'
    content = input("\n\nsliding window size (odd number) : \n")
    w_smth = int(content)
    listfile = os.listdir(pssmdir)
    print(listfile)
    for eachfile in listfile:
        # eachfile = str(i) + '.pssm'
        fi = pssmdir + '/' + eachfile  # input
        output_smth = swdir + '/' + eachfile
        output_PSSM = newdir + '/' + eachfile
        output_wop = wopdir + '/' + eachfile
        outfile_lfc = lfcdir + '/' + eachfile
        outfile_Ent = Entdir + '/' + eachfile
        output_full = fulldir + '/' + eachfile
        single_file(fi, output_smth, w_smth, output_PSSM, output_wop, outfile_lfc, outfile_Ent, output_full)
    print("\n\ntask finish. Check the file path.")