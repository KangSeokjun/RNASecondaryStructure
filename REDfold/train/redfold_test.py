import numpy as np
import subprocess
import collections
import random
import time
import sys
import os
from tqdm import tqdm
import json
import csv

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
  set_gamma
  set_rho
  set_L1
  model_saved_path: directory of saved model
  test_data_path: directory of test pickle file
  test_result_path: directory to directory to save the result csv file
  
2. on this file

 best_epoch: must be decided from train result

############################################################################################
'''
    
seed = config['seed']
num_of_device = config['num_of_device']
use_device_num = config['use_device_num']
batch_n = config['batch_n']
set_gamma = config['set_gamma']
set_rho= set_gamma+0.1
set_L1= config['set_L1']

model_saved_path = config['model_ckpt_path']
test_data_path = config['test_data_path']
test_result_path = config['test_result_path']

# Best epoch must be decided
best_epoch = -1

# seed fix for reproducing
if seed != 'none':
  seed_fix(seed)
  
if not os.path.exists(test_result_path):
  os.makedirs(test_result_path)
  
RNA_SS_data = collections.namedtuple('RNA_SS_data','name length seq_hot data_pair data_seq1 data_seq2')
  
os.environ["CUDA_VISIBLE_DEVICES"] = generate_visible_device(num_of_device)

device = torch.device('cuda:{}'.format(use_device_num))

Use_gpu= torch.cuda.is_available()

print('test data loading...')

test_data= RNASSDataGenerator(test_data_path,720)
test_len= len(test_data)
test_set= Dataset_FCN(test_data)

dataloader_test= DataLoader(dataset=test_set, batch_size=batch_n, shuffle=1, num_workers=12)

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

optimizer= torch.optim.Adam(model.parameters())

# Model on GPU
if Use_gpu:
    model= model.to(device)

model_name = 'redfold_{}th_epoch.pt'.format(best_epoch)

mod_state= torch.load(os.path.join(model_saved_path,model_name), map_location=device)
model.load_state_dict(mod_state['state_dict'])

model.eval()

with open(os.path.join(test_result_path, 'redfold_result.csv'), 'w') as f:
  fw = csv.writer(f)
  fw.writerow(['seq_name','seq_len','ppv_orig','ppv_prop','sen_orig','sen_prop','f1_orig','f1_prop','acc_orig','acc_prop'])

for index, [x1, y1, L1, seq_hot,seq_name] in enumerate(tqdm(dataloader_test, desc='testing...', ascii=True)): 
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
    
    positive_predictive_value_orig, sensitivity_orig, f1_score_orig, accuracy_orig = evaluate_exact_a(y_mask.cpu(), y1.cpu())
    positive_predictive_value_prop, sensitivity_prop, f1_score_prop, accuracy_prop = evaluate_exact_a(y_mask_proposed.cpu(), y1.cpu())
    
    optimizer.zero_grad()
    
    with open(os.path.join(test_result_path, 'redfold_result.csv'), 'a') as f:
      fw = csv.writer(f)
      fw.writerow([seq_name, int(L1[0]),
                   positive_predictive_value_orig.item(), positive_predictive_value_prop.item(),
                   sensitivity_orig.item(), sensitivity_prop.item(),
                   f1_score_orig.item(), f1_score_prop.item(),
                   accuracy_orig.item(), accuracy_prop.item()])
    