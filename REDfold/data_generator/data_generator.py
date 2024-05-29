import numpy as np
import os
import _pickle as cPickle
import collections
import torch

from multiprocessing import Pool
from torch.utils import data
from collections import Counter
from random import shuffle
from itertools import product, combinations

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}

perm = list(product(np.arange(4), np.arange(4)))
perm1= list(combinations(np.arange(4),2))
perm2= list(combinations(np.arange(16),2))
perm3= [[1,3],[3,1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]

def encoding2seq(arr):
	seq = list()
	for arr_row in list(arr):
		if sum(arr_row)==0:
			seq.append('.')
		else:
			seq.append(char_dict[np.argmax(arr_row)])
	return ''.join(seq)

def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l

class RNASSDataGenerator(object):
    def __init__(self, data_dir,Lmax, split=2, upsampling=False):
        self.data_dir = data_dir
        self.split = split
        self.upsampling = upsampling
        # Load vocab explicitly when needed
        self.load_data(data_dir,Lmax)

        # Reset batch pointer to zero
        self.batch_pointer = 0

    def __len__(self):
        'Dataset Size'
        return len(self.data)


    def load_data(self, file_pick,Lmax):
        p = Pool()
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data','name length seq_hot data_pair data_seq1 data_seq2')
        with open(file_pick, 'rb') as F1:
          self.data = cPickle.load(F1,encoding='iso-8859-1')


        if self.upsampling:
            self.data = self.upsampling_data_new()

        # List with variable length
        self.data_name = np.array([instance[0] for instance in self.data])
        self.seq_length= np.array([instance[1] for instance in self.data])
        self.seq_hot= np.array([instance[2] for instance in self.data])
        self.data_y= np.array([instance[3] for instance in self.data])
        self.data_x= np.array([instance[4] for instance in self.data])
        self.data_z= np.array([instance[5] for instance in self.data])

        self.len = len(self.data)
        self.seq_max_len = Lmax

    def upsampling_data(self):
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('/')[2], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type==t)[0]
            data_list.append(data[index])
        final_d_list= list()
        # for d in data_list:
        #     index = np.random.choice(d.shape[0], max_num)
        #     final_d_list += list(d[index])
        for i in [0, 1, 5, 7]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num)
            final_d_list += list(d[index])

        for i in [2,3,4]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num*2)
            final_d_list += list(d[index])
        
        d = data_list[6]
        index = np.random.choice(d.shape[0], int(max_num/2))
        final_d_list += list(d[index])

        shuffle(final_d_list)
        return final_d_list

    def upsampling_data_new(self):
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('_')[0], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type==t)[0]
            data_list.append(data[index])
        final_d_list= list()
        for d in data_list:
            final_d_list += list(d)
            if d.shape[0] < 300:
                index = np.random.choice(d.shape[0], 300-d.shape[0])
                final_d_list += list(d[index])
            if d.shape[0] == 652:
                index = np.random.choice(d.shape[0], d.shape[0]*4)
                final_d_list += list(d[index])
        shuffle(final_d_list)
        return final_d_list

    def upsampling_data_new_addPDB(self):
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('_')[0], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type==t)[0]
            data_list.append(data[index])
        final_d_list= list()
        for d in data_list:
            final_d_list += list(d)
            if d.shape[0] < 300:
                index = np.random.choice(d.shape[0], 300-d.shape[0])
                final_d_list += list(d[index])
            if d.shape[0] == 652:
                index = np.random.choice(d.shape[0], d.shape[0]*4)
                final_d_list += list(d[index])
        shuffle(final_d_list)
        return final_d_list

    def next_batch(self, batch_size):
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        batch_x = self.data_x[bp:bp + batch_size]
        batch_y = self.data_y[bp:bp + batch_size]
        batch_seq_len = self.seq_length[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0

        yield batch_x, batch_y, batch_seq_len

    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    def next_batch_SL(self, batch_size):
        p = Pool()
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        data_y = self.data_y[bp:bp + batch_size]
        data_seq = self.data_x[bp:bp + batch_size]
        data_pairs = self.pairs[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0
        contact = np.array(list(map(self.pairs2map, data_pairs)))
        matrix_rep = np.zeros(contact.shape)
        yield contact, data_seq, matrix_rep

    def get_one_sample(self, index):

        seq_name= self.data_name[index]
        seq_hot= self.seq_hot[index]
        seq_len= self.seq_length[index]
        seq_datay= self.data_y[index]
        seq_datax= self.data_x[index]
        seq_dataz= self.data_z[index]

        return seq_name,seq_hot,seq_len,seq_datax,seq_dataz,seq_datay

    def get_one_sample_long(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = np.nonzero(self.data_x[index].sum(axis=1))[0].max()
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact= self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name


    def random_sample(self, size=1):
        # random sample one RNA
        # return RNA sequence and the ground truth contact map
        index = np.random.randint(self.len, size=size)
        data = list(np.array(self.data)[index])
        data_seq = [instance[0] for instance in data]
        data_stru_prob = [instance[1] for instance in data]
        data_pair = [instance[-1] for instance in data]
        seq = list(map(encoding2seq, data_seq))
        contact = list(map(self.pairs2map, data_pair))
        return contact, seq, data_seq

    def get_one_sample_cdp(self, index):
        data_seq = self.data_x[index]
        data_label = self.data_y[index]

        return data_seq, data_label
      
class Dataset_Cut_concat_new_canonicle(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        if torch.is_tensor(index):
          index = index.tolist()

        # Select sample
        [seq_name,seq_hot,seq_len,data_x,data_z,data_y]= self.data.get_one_sample(index)
        #contact, data_seq, matrix_rep, data_len, data_name, data_pair = self.data.get_one_sample_addpairs(index)

        L= get_cut_len(seq_len,80)
        #-MapDinucleotide
        data_fcnS1= np.zeros((10, L, L))

        for [n, cord] in enumerate(perm1):
          [i,j]= cord
          data_fcnS1[n, :seq_len, :seq_len]= np.matmul(data_x[:seq_len, i].reshape(-1, 1), data_x[:seq_len, j].reshape(1, -1))+np.matmul(data_x[:seq_len, j].reshape(-1, 1), data_x[:seq_len, i].reshape(1, -1))


        for [i,n] in enumerate(range(6,10)):
          data_fcnS1[n, :seq_len, :seq_len]= np.matmul(data_x[:seq_len, i].reshape(-1, 1), data_x[:seq_len, i].reshape(1, -1))

        #-MapTerranucleotide
        data_fcnS2= np.zeros((136, L, L))

        for [n, cord] in enumerate(perm2):
          [i,j]= cord
          data_fcnS2[n, :seq_len, :seq_len]= np.matmul(data_z[:seq_len, i].reshape(-1, 1), data_z[:seq_len, j].reshape(1, -1))+np.matmul(data_z[:seq_len, j].reshape(-1, 1), data_z[:seq_len, i].reshape(1, -1))

        for [i,n] in enumerate(range(120,136)):
          data_fcnS2[n, :seq_len, :seq_len]= np.matmul(data_z[:seq_len, i].reshape(-1, 1), data_z[:seq_len, i].reshape(1, -1))


        data_fcn2 = np.concatenate((data_fcnS1,data_fcnS2),axis=0)

        return data_fcn2, data_y, seq_len, seq_hot, seq_name