import os as os
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn

BASE1= 'AUCG'
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

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}

pair_set= {'AU','UA','CG','GC','GU','UG'}

# Traversing paths of all files and subdirectories within a directory
def list_files_in_directory(directory):
  for root, dirs, files in os.walk(directory):
    for file in files:
      file_path = os.path.join(root, file)
      yield file_path
      
def f1_loss(pred_a, true_a, device=0, eps=1e-11):
  device = torch.device("cuda:{}".format(device))
  pred_a  = -(F.relu(-pred_a+1)-1).to(device)

  true_a = true_a.unsqueeze(1).to(device)
  unfold = nn.Unfold(kernel_size=(3, 3), padding=1)
  true_a_tmp = unfold(true_a)
  w = torch.Tensor([0, 0.0, 0, 0.0, 1, 0.0, 0, 0.0, 0]).to(device)
  true_a_tmp = true_a_tmp.transpose(1, 2).matmul(w.view(w.size(0), -1)).transpose(1, 2)
  true_a = true_a_tmp.view(true_a.shape)
  true_a = true_a.squeeze(1)

  tp = pred_a*true_a
  tp = torch.sum(tp, (1,2))

  fp = pred_a*(1-true_a)
  fp = torch.sum(fp, (1,2))

  fn = (1-pred_a)*true_a
  fn = torch.sum(fn, (1,2))

  f1 = torch.div((2*tp + eps), (2*tp + fp + fn + eps))
  return (1-f1.mean()).to(device)

def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis = 1).clamp_max(1))
    seq[contact.sum(axis = 1) == 0] = -1
    return seq

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def seqout(y1,L1,seqhot):

  seq1=""
  fast=""
  y1= y1.float()

  for a1 in range(L1):
    Id1= np.nonzero(seqhot[0,a1]).item()
    seq1+=BASE1[Id1]

    Id2= np.nonzero(y1[0,a1,:L1])
    if (Id2.nelement()):
      fast+= '(' if (a1<Id2) else ')'
    else:
      fast+='.'
  seq1+="\n"

def get_ct_dict(predict_matrix,batch_num,ct_dict):
    
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict
# randomly select one sample from the test set and perform the evaluation

def evaluate_exact_a(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a) * torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    tn_map = torch.sign((1 - torch.Tensor(pred_a)) * (1 - torch.Tensor(true_a)))
    tn = tn_map.sum()
    
    sensitivity = (tp + eps) / (tp + fn + eps)
    positive_predictive_value = (tp + eps) / (tp + fp + eps)
    f1_score = (2*tp + eps) / (2*tp + fp + fn + eps)
    accuracy = (tp+tn+eps) / (tp+tn+fp+fn+eps)
    
    return positive_predictive_value, sensitivity, f1_score, accuracy
  
def seed_fix(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # more than 1 gpu 

  # fix numpy random seed
  np.random.seed(seed)

  # fix CuDNN random seed
  cudnn.benchmark = False
  cudnn.deterministic = True

  # fix python random seed
  random.seed(seed)
  
def generate_visible_device(n):
  sequence = ','.join(map(str, range(n + 1)))
  return sequence