import numpy as np
import collections
import pickle as cPickle
import json
import os


with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    
all_data_path = config['output_path']
save_path = config['save_path']
random_seed = config['random_seed']

if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(random_seed)

RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')

with open(all_data_path, 'rb') as f:
    all_data = cPickle.load(f, encoding='iso-8859-1')
    
total_length = len(all_data)
print('Total {} data exist...\n'.format(total_length))

# train:val:test = 2:1:1 ratio
train_length = int(total_length * 0.5)
val_length = int(total_length * 0.25)
test_length = total_length - train_length - val_length

print('train data: {}\nval data: {}\ntest data:{}\n'.format(train_length, val_length, test_length))

# data shuffle and split
np.random.shuffle(all_data)

train_data = all_data[:train_length]
val_data = all_data[train_length:train_length + val_length]
test_data = all_data[train_length + val_length:]

with open(os.path.join(save_path, 'e2efold_train_split_seed_{}.pickle'.format(random_seed)), "wb") as f:
        cPickle.dump(train_data, f)
print('train pickle generation done')

with open(os.path.join(save_path, 'e2efold_val_split_seed_{}.pickle'.format(random_seed)), "wb") as f:
      cPickle.dump(val_data, f)
print('val pickle generation done')

with open(os.path.join(save_path, 'e2efold_test_split_seed_{}.pickle'.format(random_seed)), "wb") as f:
      cPickle.dump(test_data, f)
print('test pickle generation done')
