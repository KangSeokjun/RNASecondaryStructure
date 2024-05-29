import _pickle as pickle
import os
import sys
import json
import torch
import collections
import csv

import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append('./..')

from model.models import ContactAttention_simple_fix_PE
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
  epoches_first: # of epoches of this stage
  first_model_ckpt_path: directory to save model and loss
  train_data_path, val_data_path: directory of train, valid pickle file

############################################################################################
'''

seed = config['seed']
num_of_device = config['num_of_device']
use_device_num = config['use_device_num']
d = config['u_net_d']
BATCH_SIZE = config['batch_size_stage_1']
up_sampling = config['up_sampling']
epoches_first = config['epoches_first']

first_model_ckpt_path = config['first_model_ckpt_path']
train_data_path = config['train_data_path']
val_data_path = config['val_data_path']

# seed fix for reproducing
if seed != 'none':
  seed_fix(seed)

if not os.path.exists(first_model_ckpt_path):
  os.makedirs(first_model_ckpt_path)
  
os.environ["CUDA_VISIBLE_DEVICES"] = generate_visible_device(num_of_device)

device = torch.device('cuda:{}'.format(use_device_num))

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

train_data = RNASSDataGenerator(train_data_path, upsampling=up_sampling)
val_data = RNASSDataGenerator(val_data_path)

seq_len_train = train_data.data_y.shape[-2]

seq_len_val = val_data.data_y.shape[-2]

params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
train_set = Dataset(train_data)
train_generator = data.DataLoader(train_set, **params)

val_set = Dataset(val_data)
val_generator = data.DataLoader(val_set, **params)

contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len_train).to(device)

u_optimizer = optim.Adam(contact_net.parameters())

pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)

# loss record with csv format
with open(os.path.join(first_model_ckpt_path, 'loss_phase1.csv'), 'w') as f:
  fw = csv.writer(f)
  fw.writerow(['train_loss','val_loss'])
    
for epoch in range(epoches_first):
  
  running_loss_train = 0
  running_loss_val = 0
  val_f1 = 0
  
  contact_net.train()
  # num_batches = int(np.ceil(train_data.len / BATCH_SIZE))
  # for i in range(num_batches):
  
  print("{}th epoch processing...".format(epoch+1))

  for index, [contacts, seq_embeddings, matrix_reps, seq_lens, data_name] in enumerate(tqdm(train_generator, desc='train data loading...', ascii=True)):
    contacts_batch = torch.Tensor(contacts.float()).to(device)
    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
    matrix_reps_batch = torch.unsqueeze(
        torch.Tensor(matrix_reps.float()).to(device), -1)

    # padding the states for supervised training with all 0s
    state_pad = torch.zeros([matrix_reps_batch.shape[0], seq_len_train, seq_len_train]).to(device)
    PE_batch = get_pe(seq_lens, seq_len_train).float().to(device)

    
    contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len_train)).to(device)
    pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)

    # Compute loss
    loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
    running_loss_train += loss_u.data

    # Optimize the model
    u_optimizer.zero_grad()
    loss_u.backward()
    u_optimizer.step()

  contact_net.eval()
  
  for index, [contacts, seq_embeddings, matrix_reps, seq_lens, data_name] in enumerate(tqdm(val_generator, desc='valid data loading...', ascii=True)):
    contacts_batch = torch.Tensor(contacts.float()).to(device)
    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
    matrix_reps_batch = torch.unsqueeze(
        torch.Tensor(matrix_reps.float()).to(device), -1)

    # padding the states for supervised training with all 0s
    state_pad = torch.zeros([matrix_reps_batch.shape[0], seq_len_train, seq_len_train]).to(device)
    PE_batch = get_pe(seq_lens, seq_len_train).float().to(device)
    
    with torch.no_grad():
      contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len_train)).to(device)
      pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)

      # Compute loss
      loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
      running_loss_val += loss_u.data
  
  torch.save(contact_net.state_dict(), os.path.join(first_model_ckpt_path, 'e2efold_{}th_epoch.pt'.format(epoch+1)))
  with open(os.path.join(first_model_ckpt_path, 'loss_phase1.csv'), 'a') as f:
    fw = csv.writer(f)
    fw.writerow([running_loss_train.item()*BATCH_SIZE/len(train_generator),running_loss_val.item()*BATCH_SIZE/len(val_generator)])