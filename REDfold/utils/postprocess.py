import torch
import math
import numpy as np
import torch.nn.functional as F

from scipy.sparse import diags
from scipy.optimize import linear_sum_assignment

def constraint_matrix_batch(x):
    """
    this function is referred from e2efold utility function, located at https://github.com/ml4bio/e2efold/tree/master/e2efold/common/utils.py
    """
    base_a= x[:, :, 0]
    base_u= x[:, :, 1]
    base_c= x[:, :, 2]
    base_g= x[:, :, 3]
    batch= base_a.shape[0]
    length= base_a.shape[1]
    au= torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua= au + torch.transpose(au, -1, -2)
    cg= torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc= cg + torch.transpose(cg, -1, -2)
    ug= torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu= ug + torch.transpose(ug, -1, -2)

    mask= au_ua + cg_gc + ug_gu

    #Mask sharp loop
    for b1 in range(batch):
      for d2 in range(1,2):
        for i in range(d2,length):
          mask[b1,i-d2,i]= 0
        for i in range(length-d2):
          mask[b1,i+d2,i]= 0

    return mask

def symmetric_a(a_hat,m1):
    a= a_hat* a_hat
    b= torch.transpose(a, -1, -2)
    a= (a+ b)/ 2
    a= a* m1
    return a

def soft_sign(x):
    k = 1
    return 1.0/(1.0+torch.exp(-2*k*x))

def postprocess_orig(u1, x1, L1, lr_min, lr_max, num_itr, rho=0.0, with_L1=False,s9=math.log(9.0)):
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
    m= constraint_matrix_batch(x1).float()
    u1= soft_sign(u1 - s9)* u1


    # Initialization
    a_hat= (torch.sigmoid(u1))* soft_sign(u1-s9).detach()
    s_hat= symmetric_a(a_hat, m)

    # constraint Y_hat
    sumcol_hat= torch.sum(s_hat, dim=-1)
    lambd= F.relu(sumcol_hat- 1).detach()


    # gradient descent approach
    for t in range(num_itr):
      s_hat= symmetric_a(a_hat, m)
      sumcol_hat= torch.sum(s_hat, dim=-1)
      grad_a= (lambd * soft_sign(sumcol_hat- 1)).unsqueeze_(-1).expand(u1.shape)-u1/2
      grad= a_hat* m* (grad_a + torch.transpose(grad_a, -1, -2))
      a_hat-= lr_min * grad
      a_hat= F.relu(a_hat)
      lr_min*= 0.99

      if with_L1:
        a_hat= F.relu(torch.abs(a_hat) - rho* lr_min)

      lambd_grad= F.relu(torch.sum( symmetric_a(a_hat, m),dim=-1)- 1 )
      lambd+= lr_max * lambd_grad
      lr_max*= 0.99

    # Constraint A+AT
    ya= symmetric_a(a_hat,m)
    s2=torch.squeeze(torch.sum((ya>0.5),1))
    
    # Find single row
    for a1 in range(L1):
      if (s2[a1]>1):
        Id1= torch.nonzero(ya[0,a1,:])
        Idm= torch.argmax(ya[0,a1,:])
        for a2 in Id1:
          if not (a2==Idm):
            ya[0,a1,a2]=0
            ya[0,a2,a1]=0
    
    ya = (ya > 0.5).type(torch.cuda.FloatTensor)
    
    result = ya.cpu().numpy()
    return ya

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
    m1 = au + gc + gu
    
    m2 = (u>s)
    
    m3 = 1 - diags([1]*7, [-3, -2, -1, 0, 1, 2, 3], shape=(u.shape[-2], u.shape[-1])).toarray()
    m3 = torch.Tensor(m3).to(device)
    
    return m1*m2*m3

def postprocess_proposed(u, x, s=math.log(9.0), process_device=0, rho=0):
  m = constraint_matrix_proposed(u, x, s, process_device)
  
  cur_u = (u-s) * m
  
  batch_size = x.shape[0]
  
  a = torch.zeros_like(u)
  
  for batch in range(batch_size):
      row_ind, col_ind = linear_sum_assignment(- cur_u[batch].cpu())
      a[batch, row_ind, col_ind] = 1
  
  a = a * m
  a = a + torch.transpose(a, -1, -2)              
  
  result = a.cpu().numpy()
  
  result[result>0] = 1
  
  # print(result)
  
  re = torch.tensor(result)
  
  device = torch.device('cuda:{}'.format(process_device))
  
  return re.to(device).type(torch.cuda.FloatTensor)