import _pickle as pickle

import os
import sys
import json
import collections
import csv
import torch
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append('./..')

from model.models import ContactAttention_simple_fix_PE, Lag_PP_mixed, RNA_SS_e2e
from utils.utils import *
from data_generator.data_generator import RNASSDataGenerator, Dataset
from utils.postprocess import postprocess, postprocess_proposed

with open("config.json", "r", encoding="utf-8") as f:
  config = json.load(f)
  
'''
############################################################################################

Parameter you can change

############################################################################################

1. on 'config.json'

  seed: random seed to fix for reproduction (set 'none' to avoide fixing the random seed)
  num_of_device: # of gpu
  use_device_num: gpu device number (obtained using 'nvidia-smi') you want to use
  d
  pp_steps
  k
  third_model_ckpt_path: directory of saved model
  test_data_path: directory of test pickle file
  test_result_path: directory to save the result csv file
  
2. on this file

 BATCH_SIZE
 best_epoch: must be decided from stage3 result
 s

############################################################################################
'''

seed = config['seed']
num_of_device = config['num_of_device']
use_device_num = config['use_device_num']
d = config['u_net_d']
BATCH_SIZE = 1
pp_steps = config['pp_steps']
k = config['k']
s = math.log(9.0)

third_model_ckpt_path = config['third_model_ckpt_path']
test_result_path = config['test_result_path']
test_data_path = config['test_data_path']

# Best epoch must be decided
best_epoch = -1

if not os.path.exists(test_result_path):
    os.makedirs(test_result_path)
    
# seed fix for reproducing
if seed != 'none':
  seed_fix(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = generate_visible_device(num_of_device)

device = torch.device('cuda:{}'.format(use_device_num))

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

test_data = RNASSDataGenerator(test_data_path)

seq_len = test_data.data_y.shape[-2]

params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}

test_set = Dataset(test_data)
test_generator = data.DataLoader(test_set, **params)

contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len).to(device)
lag_pp_net = Lag_PP_mixed(pp_steps, k, device=use_device_num).to(device)
rna_ss_e2e = RNA_SS_e2e(contact_net.to(device), lag_pp_net.to(device)).to(device)
rna_ss_e2e.load_state_dict(torch.load(os.path.join(third_model_ckpt_path, 'e2efold_{}th_epoch.pt'.format(best_epoch)), map_location = device))
rna_ss_e2e.to(device)

contact_net = rna_ss_e2e.model_att
lag_pp_net = rna_ss_e2e.model_pp

contact_net.eval()
lag_pp_net.eval()
rna_ss_e2e.eval()

for index, [contacts, seq_embeddings, matrix_reps, seq_lens, data_name] in enumerate(tqdm(test_generator, desc='test data loading...', ascii=True)):
    
  contacts_batch = torch.Tensor(contacts.float()).to(device)
  seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
  matrix_reps_batch = torch.unsqueeze(
      torch.Tensor(matrix_reps.float()).to(device), -1)

  state_pad = torch.zeros(contacts.shape).to(device)

  PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
  
  # In this script, individual metric for each data is saved in csv format
  if index == 0:
    with open(os.path.join(test_result_path, 'result.csv'), 'w') as f:
      fw = csv.writer(f)
      fw.writerow(['seq_name','seq_len','ppv_orig','ppv_prop','sen_orig','sen_prop','f1_orig','f1_prop','acc_orig','acc_prop'])
      
  
  with torch.no_grad():
    pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, seq_embedding_batch, state_pad)
    pred_contacts_sigmoid = torch.sigmoid(pred_contacts)
    
    u_no_train = postprocess(pred_contacts, seq_embedding_batch, 0.01, 0.1, 50, 1.0, True, s=s)
    map_no_train = (u_no_train > 0.5).float()
    
    positive_predictive_value, sensitivity, f1_score, accuracy = evaluate_exact_a(map_no_train.cpu(), contacts_batch.cpu())

    u_no_train_2 = postprocess_proposed(pred_contacts[:,:seq_lens, :seq_lens], seq_embedding_batch[:,:seq_lens,:seq_lens],s=s, process_device=use_device_num)
    
    positive_predictive_value_prop, sensitivity_prop, f1_score_prop, accuracy_prop = evaluate_exact_a(u_no_train_2.cpu(), contacts_batch[:,:seq_lens,:seq_lens].cpu())

    with open(os.path.join(test_result_path, 'result.csv'), 'a') as f:
      fw = csv.writer(f)
      fw.writerow([data_name, seq_lens.item(), positive_predictive_value.item(), positive_predictive_value_prop.item(), sensitivity.item(), sensitivity_prop.item(),
                    f1_score.item(), f1_score_prop.item(), accuracy.item(), accuracy_prop.item()])