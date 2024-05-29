import numpy as np
import math

import torch
import torch.nn.functional as F

from scipy.sparse import diags
from scipy.optimize import linear_sum_assignment

def contact_a(a_hat, m):
  a = a_hat * a_hat
  a = (a + torch.transpose(a, -1, -2)) / 2
  a = a * m
  return a

def soft_sign(x):
  k = 1
  return 1.0/(1.0+torch.exp(-2*k*x))

def constraint_matrix_batch(x):
  base_a = x[:, :, 0]
  base_u = x[:, :, 1]
  base_c = x[:, :, 2]
  base_g = x[:, :, 3]
  batch = base_a.shape[0]
  length = base_a.shape[1]
  au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
  au_ua = au + torch.transpose(au, -1, -2)
  cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
  cg_gc = cg + torch.transpose(cg, -1, -2)
  ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
  ug_gu = ug + torch.transpose(ug, -1, -2)
  return au_ua + cg_gc + ug_gu

def constraint_matrix_proposed(u, x, s, process_device):
  device = torch.device('cuda:{}'.format(process_device))

  base_a = x[:, :, 0]
  base_u = x[:, :, 1]
  base_c = x[:, :, 2]
  base_g = x[:, :, 3]
  batch = base_a.shape[0]
  length = base_a.shape[1]
  base_ag = base_a + base_g
  base_uc = base_u + base_c
  ag_uc = torch.matmul(base_ag.view(batch, length, 1), base_uc.view(batch, 1, length))
  
  au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
  gc = torch.matmul(base_g.view(batch, length, 1), base_c.view(batch, 1, length))
  gu = torch.matmul(base_g.view(batch, length, 1), base_u.view(batch, 1, length))
  m1 = (au + gc + gu).to(device)
  
  m2 = (u>s).to(device)
  
  m3 = 1 - diags([1]*7, [-3, -2, -1, 0, 1, 2, 3], shape=(u.shape[-2], u.shape[-1])).toarray()
  m3 = torch.Tensor(m3).to(device)
  
  # print('u.shape[-2]: {}, u.shape[-1]: {}, m3.shape:{}\n'.format(u.shape[-2], u.shape[-1],m3.shape))
  
  return m1*m2*m3

def postprocess(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=math.log(9.0)):
  """
  :param u: utility matrix, u is assumed to be symmetric, in batch
  :param x: RNA sequence, in batch
  :param lr_min: learning rate for minimization step
  :param lr_max: learning rate for maximization step (for lagrangian multiplier)
  :param num_itr: number of iterations
  :param rho: sparsity coefficient
  :param with_l1:
  :return:
  """
  m = constraint_matrix_batch(x)
  # u with threshold
  # equivalent to sigmoid(u) > 0.9
  # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
  u = soft_sign(u - s) * u

  # initialization
  a_hat = (torch.sigmoid(u)) * soft_sign(u - math.log(9.0)).detach()
  lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

  # gradient descent
  for t in range(num_itr):

    grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
    grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
    a_hat -= lr_min * grad
    lr_min = lr_min * 0.99

    if with_l1:
        a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

    lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
    lmbd += lr_max * lmbd_grad
    lr_max = lr_max * 0.99

  a = a_hat * a_hat
  a = (a + torch.transpose(a, -1, -2)) / 2
  a = a * m
  return a

def postprocess_proposed(u, x, s=math.log(9.0), process_device=0):
  device = torch.device('cuda:{}'.format(process_device)) 
  m = constraint_matrix_proposed(u, x, s, process_device)
  
  cur_u = (u-s).to(device) * m
  
  batch_size = x.shape[0]
  
  a = torch.zeros_like(u).to(device)
  
  for batch in range(batch_size):
      row_ind, col_ind = linear_sum_assignment(- cur_u[batch].cpu())
      a[batch, row_ind, col_ind] = 1
  
  a = a * m
  a = a + torch.transpose(a, -1, -2)              
  
  result = a.cpu().numpy()
  
  result[result>0] = 1
  
  # print(result)
  
  device = torch.device('cuda:{}'.format(process_device))
  
  re = torch.tensor(result)
  
  return re.to(device).type(torch.cuda.FloatTensor)