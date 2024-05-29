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
  up_sampling: When using the RNAStrAlign dataset, set this value to true
  d
  BATCH_SIZE
  pp_steps
  step_gamma
  k
  epoches_first: # of epoches of previous stage
  epoches_third: # of epoches of this stage
  first_model_ckpt_path: directory of saved model of previous stage
  third_model_ckpt_path: directory to save model and loss
  train_data_path, val_data_path: directory of train, valid pickle file

############################################################################################
'''

seed = config['seed']
num_of_device = config['num_of_device']
use_device_num = config['use_device_num']
d = config['u_net_d']
BATCH_SIZE = config['batch_size_stage_3']
up_sampling = config['up_sampling']
epoches_first = config['epoches_first']
epoches_third = config['epoches_third']
pp_steps = config['pp_steps']
step_gamma = config['step_gamma']
k = config['k']

first_model_ckpt_path = config['first_model_ckpt_path']
third_model_ckpt_path = config['third_model_ckpt_path']
train_data_path = config['train_data_path']
val_data_path = config['val_data_path']

# seed fix for reproducing
if seed != 'none':
  seed_fix(seed)
  
if not os.path.exists(first_model_ckpt_path):
  os.makedirs(first_model_ckpt_path)

if not os.path.exists(third_model_ckpt_path):
  os.makedirs(third_model_ckpt_path)
  
os.environ["CUDA_VISIBLE_DEVICES"] = generate_visible_device(num_of_device)

device = torch.device('cuda:{}'.format(use_device_num))

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

train_data = RNASSDataGenerator(train_data_path, upsampling=up_sampling)
val_data = RNASSDataGenerator(val_data_path)

seq_len = train_data.data_y.shape[-2]

params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
train_set = Dataset(train_data)
train_generator = data.DataLoader(train_set, **params)

val_set = Dataset(val_data)
val_generator = data.DataLoader(val_set, **params)

contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len).to(device)
contact_net.load_state_dict(torch.load(os.path.join(first_model_ckpt_path,'e2efold_{}th_epoch.pt'.format(epoches_first)), map_location=device))
lag_pp_net = Lag_PP_mixed(pp_steps, k, use_device_num).to(device)

rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)

all_optimizer = optim.Adam(rna_ss_e2e.parameters())

pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight = pos_weight)
criterion_mse = torch.nn.MSELoss(reduction='sum')

steps_done = 0

# loss record with csv format
with open(os.path.join(third_model_ckpt_path,'loss_phase3.csv'), 'w') as f:
  fw = csv.writer(f)
  fw.writerow(['train_loss','val_loss'])
  
all_optimizer.zero_grad()

for epoch in range(epoches_third):
    running_loss_train = 0
    running_loss_val = 0
    
    rna_ss_e2e.train()
    print('{}th epoch processing...'.format(epoch+1))
    for index, [contacts, seq_embeddings, matrix_reps, seq_lens, data_name] in enumerate(tqdm(train_generator, desc='train data loading...', ascii=True)):
      
      contacts_batch = torch.Tensor(contacts.float()).to(device)
      seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
      matrix_reps_batch = torch.unsqueeze(
          torch.Tensor(matrix_reps.float()).to(device), -1)
      
      contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
      # padding the states for supervised training with all 0s
      state_pad = torch.zeros([matrix_reps_batch.shape[0], 
          seq_len, seq_len]).to(device)

      PE_batch = get_pe(seq_lens, seq_len).float().to(device)
      # the end to end model
      pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, 
          seq_embedding_batch, state_pad)

      loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

      # Compute loss, consider the intermidate output
      loss_a  = f1_loss(a_pred_list[-1]*contact_masks, contacts_batch, device= use_device_num, eps=1e-11).to(device)
      
      for i in range(pp_steps-1):
        loss_a += np.power(step_gamma, pp_steps-1-i)*f1_loss(a_pred_list[i]*contact_masks, contacts_batch, device=use_device_num, eps=1e-11)
          
      mse_coeff = 1.0/pp_steps

      loss_a = mse_coeff*loss_a.to(device)

      loss = loss_u + loss_a
      running_loss_train += loss.data

      # Optimize the model, we increase the batch size by 100 times
      loss.backward()
      if steps_done % 30 ==0:
        all_optimizer.step()
        all_optimizer.zero_grad()
      steps_done=steps_done+1

    torch.save(rna_ss_e2e.state_dict(), os.path.join(third_model_ckpt_path, 'e2efold_{}th_epoch.pt'.format(epoch+1)))
    
    rna_ss_e2e.eval()
    for index, [contacts, seq_embeddings, matrix_reps, seq_lens, data_name] in enumerate(tqdm(val_generator, desc='valid data loading...', ascii=True)):
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        matrix_reps_batch = torch.unsqueeze(
            torch.Tensor(matrix_reps.float()).to(device), -1)
        
        contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
        # padding the states for supervised training with all 0s
        state_pad = torch.zeros([matrix_reps_batch.shape[0], seq_len, seq_len]).to(device)

        PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        
        with torch.no_grad():
          # the end to end model
          pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, seq_embedding_batch, state_pad)

          loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

          # Compute loss, consider the intermidate output
          loss_a  = f1_loss(a_pred_list[-1]*contact_masks, contacts_batch, device=use_device_num, eps=1e-11).to(device)
    
          for i in range(pp_steps-1):
            loss_a += np.power(step_gamma, pp_steps-1-i)*f1_loss(a_pred_list[i]*contact_masks, contacts_batch, device=use_device_num, eps=1e-11)
              
          mse_coeff = 1.0/pp_steps

          loss_a = mse_coeff*loss_a.to(device)

          loss = loss_u + loss_a
          
          running_loss_val += loss.data
          
    with open(os.path.join(third_model_ckpt_path, 'loss_phase3.csv'), 'a') as f:
      fw = csv.writer(f)
      fw.writerow([running_loss_train.item()*BATCH_SIZE/len(train_generator), running_loss_val.item()*BATCH_SIZE/len(val_generator)])