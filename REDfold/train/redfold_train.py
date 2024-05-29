import numpy as np
import subprocess
import collections
import random
import time
import sys
import os
from tqdm import tqdm
import json

#from network import FCDenseNet as Model
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append('./..')

from utils.utils import *
from data_generator.data_generator import RNASSDataGenerator
from data_generator.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from model.models import FCDenseNet
from utils.postprocess import postprocess_orig, postprocess_proposed

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
  batch_n: BATCH_SIZE
  epoch_n: # of epoches
  set_gamma
  set_rho
  set_L1
  model_ckpt_path: directory to save model and loss
  train_data_path, val_data_path: directory of train, valid pickle file

############################################################################################
'''
    
seed = config['seed']
num_of_device = config['num_of_device']
use_device_num = config['use_device_num']
batch_n = config['batch_n']
epoch_n = config['epoch_n']
set_gamma = config['set_gamma']
set_rho = set_gamma+0.1
set_L1 = config['set_L1']

model_ckpt_path = config['model_ckpt_path']
train_data_path = config['train_data_path']
val_data_path = config['val_data_path']

# seed fix for reproducing
if seed != 'none':
  seed_fix(seed)
  
if not os.path.exists(model_ckpt_path):
  os.makedirs(model_ckpt_path)
  
RNA_SS_data = collections.namedtuple('RNA_SS_data','name length seq_hot data_pair data_seq1 data_seq2')
  
os.environ["CUDA_VISIBLE_DEVICES"] = generate_visible_device(num_of_device)

device = torch.device('cuda:{}'.format(use_device_num))

Use_gpu= torch.cuda.is_available()

#positive set balance weight for loos function
loss_weight= torch.Tensor([300]).to(device)

print('train data loading...')

train_data= RNASSDataGenerator(train_data_path,720)
train_len= len(train_data)
train_set= Dataset_FCN(train_data)

dataloader_train= DataLoader(dataset=train_set, batch_size=batch_n, shuffle=1, num_workers=12)

print('valid data loading...')

valid_data= RNASSDataGenerator(val_data_path,720)
valid_len= len(valid_data)
valid_set= Dataset_FCN(valid_data)

dataloader_valid= DataLoader(dataset=valid_set, batch_size=batch_n, shuffle=1, num_workers=12)

#- Network
model= FCDenseNet(in_channels=146,out_channels=1,
                initial_num_features=16,
                dropout=0,

                down_dense_growth_rates=(4,8,16,32),
                down_dense_bottleneck_ratios=None,
                down_dense_num_layers=(4,4,4,4),
                down_transition_compression_factors=1.0,

                middle_dense_growth_rate=32,
                middle_dense_bottleneck=None,
                middle_dense_num_layers=8,

                up_dense_growth_rates=(64,32,16,8),
                up_dense_bottleneck_ratios=None,
                up_dense_num_layers=(4,4,4,4))

# Model on GPU
if Use_gpu:
    model= model.to(device)

loss_f= torch.nn.BCEWithLogitsLoss(pos_weight= loss_weight)
optimizer= torch.optim.Adam(model.parameters(), lr=1e-3)

# val_loss_orig: Validation loss using the existing post-processing
# val_loss_prop: Validation loss using the our post-processing
with open(os.path.join(model_ckpt_path, 'loss.txt'), 'w') as fr:
    fr.write('train_loss,val_loss_orig,val_loss_prop\n')
    
for epoch in range(epoch_n):
    running_loss_train= 0
    running_loss_val_orig = 0
    running_f1_loss_val_orig = 0
    running_loss_val_prop = 0
    running_f1_loss_val_prop = 0
    print(f"-"*10)
    print(f"Epoch {epoch+1}/{epoch_n}")
    print(f"-"*10)
    print("Phase train...")

    model.train()
        
    print(f"Train data:{len(dataloader_train)}")

    for index, [x1, y1, L1, seq_hot,seq_name] in enumerate(tqdm(dataloader_train, desc='training...', ascii=True)):
        # Data on GPU
        if Use_gpu:
            x1= x1.to(device).type(torch.cuda.FloatTensor)
            y1= y1.to(device).type(torch.cuda.FloatTensor)

        [x1, y1]= Variable(x1), Variable(y1)

        y_pred= model(x1)
        # Mask Matrix
        mask1= torch.zeros_like(y_pred)
        mask1[:, :L1,:L1] = 1

        y_mask= y_pred*mask1
        
        optimizer.zero_grad()
        loss_train= loss_f(y_mask, y1)
        
        loss_train.backward()
        optimizer.step()

        running_loss_train+= loss_train.item()

    epoch_loss_train= running_loss_train*batch_n/train_len
    
    print(f"Epoch Loss Train:{epoch_loss_train:.4f}")

    # 모델 저장
    model_name = 'redfold_{}th_epoch.pt'.format(epoch+1)
    mod_state= {'epoch': epoch+1, 'state_dict': model.state_dict()}
    torch.save(mod_state, os.path.join(model_ckpt_path, model_name))
        
    print(f"Validation data:{len(dataloader_valid)}")

    model.eval()
    
    for index, [x1, y1, L1, seq_hot,seq_name] in enumerate(tqdm(dataloader_valid, desc='validating...', ascii=True)): 
        # Data on GPU
        if Use_gpu:
            x1= x1.to(device).type(torch.cuda.FloatTensor)
            y1= y1.to(device).type(torch.cuda.FloatTensor)

        [x1, y1]= Variable(x1), Variable(y1)
        
        with torch.no_grad():
            y_pred= model(x1)
        
        # post-processing without learning train
        seq_hot=seq_hot.to(device)
        y_mask= postprocess_orig(y_pred,seq_hot,L1, 0.01, 0.1, 100, set_rho,set_L1,set_gamma)
        y_mask_proposed = postprocess_proposed(y_pred, seq_hot, set_gamma, use_device_num)
        
        optimizer.zero_grad()
        
        f1_loss_val_orig = f1_loss(y_mask, y1, device=use_device_num)
        f1_loss_val_prop = f1_loss(y_mask_proposed, y1, device=use_device_num)
        
        running_f1_loss_val_orig+= f1_loss_val_orig.item()
        running_f1_loss_val_prop+= f1_loss_val_prop.item()
    
    epoch_f1_loss_val_orig = running_f1_loss_val_orig*batch_n/valid_len
    epoch_f1_loss_val_prop = running_f1_loss_val_prop*batch_n/valid_len
    
    with open(os.path.join(model_ckpt_path, 'loss.txt'), 'a') as fr:
        fr.write('{},{},{}\n'.format(epoch_loss_train,epoch_f1_loss_val_orig,epoch_f1_loss_val_prop))