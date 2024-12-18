import numpy as np
import torch
import math
from tqdm import tqdm
from code_generation import complementary, random_pair_generator, RM_encoder, accuracy

class LossF:
  def __init__(self, H, y, la):
    self.H = H
    self.y = y
    self.la = la

  def __call__(self, x):
    p1_1 = (self.H @ x) % 2
    p1_2 = 2 - p1_1
    p1 = p1_1 - torch.nn.functional.relu(p1_1 - p1_2)
    p2 = (x - self.y) % 2
    return p1.norm()  +  self.la * p2.norm()

def optimize(H, y):
  x = y.clone().detach().requires_grad_(True)
  loss_f = LossF(H, y, 0.)
  min_loss = np.inf
  min_vec = None
  optimizer = torch.optim.SGD([x], lr=6*1e-4)

  for i in range(6000):
    loss = loss_f(x)
    #print(loss.item())
    loss.backward()
    #print(x.grad)
    optimizer.step()
    optimizer.zero_grad()
    if loss < min_loss:
      min_loss = loss
      min_vec = x
  return min_vec

def measure_different(m, r, epsilon):
    times = 1000
    H_matrix = complementary(m, r)
    H_matrix = torch.tensor(H_matrix)
    anses = []
    for i in tqdm(range(times)):
        default, noised = random_pair_generator(m, r, epsilon)
        real_ans = torch.tensor(RM_encoder(m, r, default) )
        noised = torch.tensor(noised)
        decoded = optimize(H_matrix, noised)
        decoded_rounded =  torch.round(decoded) % 2
        anses.append((real_ans, decoded, decoded_rounded))
    w = []
    for l1, _, l2 in anses:
        w.append(accuracy(l1, l2))
    return np.mean(w)

values = []
mrs = [(4,1,0.1),(4,1,0.12),(4,1,0.15),(4,1,0.2),(4,1,0.04),(5,1,0.2)]

for m, r, epsilon in mrs:
    acc = measure_different(m, r, epsilon)
    values.append(acc)

print(mrs)
print(values)

