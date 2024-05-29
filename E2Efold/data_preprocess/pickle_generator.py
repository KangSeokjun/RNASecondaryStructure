import os
import sys
import _pickle as cPickle
import numpy as np
import json
import pandas as pd
import collections

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append('./..')

from utils.utils import *


label_dict = {
    '.': np.array([1,0,0]), 
    '(': np.array([0,1,0]), 
    ')': np.array([0,0,1])
}
seq_dict = {
    'A':np.array([1,0,0,0]),
    'U':np.array([0,1,0,0]),
    'C':np.array([0,0,1,0]),
    'G':np.array([0,0,0,1]),
    'N':np.array([0,0,0,0])
}

def generate_label(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] <= 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)

def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)

def stru_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: label_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)

def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0,maxlen-a),(0,0)), 'constant')
  
  

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# directory containing the data-generated .ct files
ct_files_path = config['ct_files_path']
length_limit = config['length_limit']

file_list = list()

for file in list_files_in_directory(ct_files_path):
  file_list.append(file)


data_list = list(map(lambda x: pd.read_csv(x, sep='\s+', skiprows=1,
        header=None), file_list))
seq_len_list= list(map(len, data_list))

file_length_dict = dict()
for i in range(len(seq_len_list)):
    file_length_dict[file_list[i]] = seq_len_list[i]
    
data_list = list(filter(lambda x: len(x)<=length_limit, data_list))
seq_len_list = list(map(len, data_list))
file_list = list(filter(lambda x: file_length_dict[x]<=length_limit, file_list))
pairs_list = list(map(get_pairings, data_list))

structure_list = list(map(generate_label, data_list))
seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])), data_list))

# label and sequence encoding, plus padding to the maximum length
seq_encoding_list = list(map(seq_encoding, seq_list))
stru_encoding_list = list(map(stru_encoding, structure_list))

seq_encoding_list_padded = list(map(lambda x: padding(x, length_limit), 
    seq_encoding_list))
stru_encoding_list_padded = list(map(lambda x: padding(x, length_limit), 
    stru_encoding_list))

# gather the information into a list of tuple
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')
RNA_SS_data_list = list()
for i in range(len(data_list)):
    RNA_SS_data_list.append(RNA_SS_data(seq=seq_encoding_list_padded[i],
        ss_label=stru_encoding_list_padded[i], 
        length=seq_len_list[i], name=file_list[i], pairs=pairs_list[i]))


outputpath = config['output_path']
with open(outputpath, 'wb') as f:
  cPickle.dump(RNA_SS_data_list, f)