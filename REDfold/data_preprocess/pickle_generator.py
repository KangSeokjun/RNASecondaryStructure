import numpy as np
import subprocess
import collections
import pickle as cPickle
import random
import time
import sys
import os
import json

from Bio import SeqIO
from itertools import product, combinations

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append('./..')

from utils.utils import *

BASE1 = 'AUCG'
pair_set= {'AU','UA','CG','GC','GU','UG'}

global npBASE1
global dcBASE2

def one_hot(seq1):
    RNA_seq= seq1

    feat= np.concatenate([[(npBASE1 == base.upper()).astype(int)] 
          if str(base).upper() in BASE1 else np.array([[0] * len(BASE1)]) for base in RNA_seq])

    return feat


def one_hot_2m(seq1):
    L1= len(seq1)
    feat= np.zeros((L1,16))
    for i in range(0,L1-1):
      Id1= str(seq1[i:i+2]).upper()
      if Id1 in dcBASE2:
        feat[i,dcBASE2[Id1]]= 1
    #Circle Back 2mer
    Id1= str(seq1[-1]+seq1[0]).upper()
    feat[L1-1,dcBASE2[Id1]]= 1

    return feat




def get_cut_len(data_len,set_len):
    L= data_len
    if L<= set_len:
        L= set_len
    else:
        L= (((L - 1) // 16) + 1) * 16
    return L


#- Check standard pairs
def check_stand(pairs, seq):
  for pair in pairs:
    str1= seq[pair[0]]+seq[pair[1]]
    if (str1 not in pair_set):
      print(f"Error: Pair({pair})->{str1}")
      return False
      
  return True


def pair2map(pairs, seq_len):
  pmap= np.zeros([seq_len, seq_len])
  for pair in pairs:
    pmap[pair[0], pair[1]] = 1
  return pmap

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

RNA_SS_data = collections.namedtuple('RNA_SS_data','name length seq_hot data_pair data_seq1 data_seq2')

# irectory containing the data-generated .ct files
ct_files_path = config['ct_files_path']
length_limit = config['length_limit']

# RNA data Container
all_files_list = []

npBASE1= np.array([b1 for b1 in BASE1])
npBASE2= np.array(["".join(b2) for b2 in product(npBASE1,npBASE1)])
dcBASE2= {}
for [a,b] in enumerate(npBASE2):
  dcBASE2[b]= a
  
all_files= os.listdir(ct_files_path)

for index,item_file in enumerate(all_files):
  t0= subprocess.getstatusoutput('awk \'{print $2}\' '+os.path.join(ct_files_path,item_file))
  t0 = subprocess.getstatusoutput("awk 'NR > 1 {print $2}' " +os.path.join(ct_files_path,item_file))
  t1= subprocess.getstatusoutput("awk 'NR > 1 {print $1}' "+os.path.join(ct_files_path,item_file))
  t2= subprocess.getstatusoutput("awk 'NR > 1 {print $5}' "+os.path.join(ct_files_path,item_file))
  seq= ''.join(t0[1].split('\n'))
  
  if t0[0] == 0:
    try:
      one_hot_matrix= one_hot(seq.upper())
      one_hot_mat2= one_hot_2m(seq.upper())
    except IndexError as ie:
      print(f"{ie}")
  
  if t1[0] == 0 and t2[0] == 0:
    pair_dict_all_list = [[int(item_tmp)-1,int(t2[1].split('\n')[index_tmp])-1] for index_tmp,item_tmp in enumerate(t1[1].split('\n')) if int(t2[1].split('\n')[index_tmp]) != 0]
  else:
    pair_dict_all_list = []
    
  n_item= item_file.rfind('.')
  if n_item!=-1:
    seq_name= item_file[:n_item]
  else:
    seq_name = item_file
    
  seq_len = len(seq)
  
  pair_dict_all = dict([item for item in pair_dict_all_list if item[0]<item[1]])
  
  if not (check_stand(pair_dict_all_list,seq)):
    exit()

  if index%1==0:
    print('current processing %d/%d'%(index+1,len(all_files)), end='\r')
    
  ss_label = np.zeros((seq_len,3),dtype=int)
  ss_label[[*pair_dict_all.keys()],] = [0,1,0]
  ss_label[[*pair_dict_all.values()],] = [0,0,1]

  L= get_cut_len(seq_len,80)
  
  ##-Trans seq to seq_length
  one_hot_matrix_LM= np.zeros((L,4))
  one_hot_matrix_LM[:seq_len,]= one_hot_matrix
  # ss_label_L= np.zeros((L,3),dtype=int)

  one_hot_mat2_LM= np.zeros((L,16))
  one_hot_mat2_LM[:seq_len,]= one_hot_mat2
  
  data_seq1= one_hot_matrix_LM
  data_seq2= one_hot_mat2_LM
  
  ##-Seq_onehot
  seq_hot= one_hot_matrix_LM[:L,:]
  data_pair= pair2map(pair_dict_all_list,L)
  
  sample_tmp= RNA_SS_data(name=seq_name, length=seq_len, seq_hot=seq_hot, data_pair=data_pair, data_seq1= data_seq1, data_seq2=data_seq2)      
  all_files_list.append(sample_tmp)

outputpath = config['output_path']
with open(outputpath, 'wb') as f:
  cPickle.dump(all_files_list, f)