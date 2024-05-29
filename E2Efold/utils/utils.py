import os as os
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}

# Traversing paths of all files and subdirectories within a directory
def list_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            yield file_path
            
# return index of contact pairing, index start from 0
def get_pairings(data):
    rnadata1 = list(data.loc[:,0].values)
    rnadata2 = list(data.loc[:,4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1]>0, rna_pairs))
    rna_pairs = (np.array(rna_pairs)-1).tolist()
    return rna_pairs
  
def encoding2seq(arr):
	seq = list()
	for arr_row in list(arr):
		if sum(arr_row)==0:
			seq.append('.')
		else:
			seq.append(char_dict[np.argmax(arr_row)])
	return ''.join(seq)

def get_pe(seq_lens, max_len):
  num_seq = seq_lens.shape[0]
  pos_i_abs = torch.Tensor(np.arange(1,max_len+1)).view(1, 
      -1, 1).expand(num_seq, -1, -1).double()
  pos_i_rel = torch.Tensor(np.arange(1,max_len+1)).view(1, -1).expand(num_seq, -1)
  pos_i_rel = pos_i_rel.double()/seq_lens.view(-1, 1).double()
  pos_i_rel = pos_i_rel.unsqueeze(-1)
  pos = torch.cat([pos_i_abs, pos_i_rel], -1)

  PE_element_list = list()
  # 1/x, 1/x^2
  PE_element_list.append(pos)
  PE_element_list.append(1.0/pos_i_abs)
  PE_element_list.append(1.0/torch.pow(pos_i_abs, 2))

  # sin(nx)
  for n in range(1, 50):
    PE_element_list.append(torch.sin(n*pos))

  # poly
  for i in range(2, 5):
    PE_element_list.append(torch.pow(pos_i_rel, i))

  for i in range(3):
    gaussian_base = torch.exp(-torch.pow(pos, 
      2))*math.sqrt(math.pow(2,i)/math.factorial(i))*torch.pow(pos, i)
    PE_element_list.append(gaussian_base)

  PE = torch.cat(PE_element_list, -1)
  for i in range(num_seq):
      PE[i, seq_lens[i]:, :] = 0
  return PE
  
def contact_map_masks(seq_lens, max_len):
  n_seq = len(seq_lens)
  masks = np.zeros([n_seq, max_len, max_len])
  for i in range(n_seq):
    l = int(seq_lens[i].cpu().numpy())
    masks[i, :l, :l]=1
  return masks
  
# for test the f1 loss filter
# true_a = torch.Tensor(np.arange(25)).view(5,5).unsqueeze(0)

def f1_loss(pred_a, true_a, device=0, eps=1e-11 ):
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

def evaluate_exact_a(pred_a, true_a, eps=1e-11):
  tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
  tp = tp_map.sum()
  pred_p = torch.sign(torch.Tensor(pred_a)).sum()
  true_p = true_a.sum()
  fp = pred_p - tp
  fn = true_p - tp
  tn_map = torch.sign((1- torch.Tensor(pred_a)) * (1 - torch.Tensor(true_a)))
  tn = tn_map.sum()
  
  sensitivity = (tp + eps) / (tp + fn + eps)
  positive_predictive_value = (tp + eps) / (tp + fp + eps)
  f1_score = (2*tp + eps) / (2*tp + fp + fn + eps)
  accuracy = (tp+tn+eps) / (tp+tn+fp+fn+eps)
  return positive_predictive_value, sensitivity, f1_score, accuracy