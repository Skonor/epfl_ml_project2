import numpy as np
import math


def num_to_binom(n,m):
  '''Takes an integer n as an input, and outputs the binomial representation array b of n as an output.
     Here, n is an integer, b is an array and m is the length of b satisfying 2^m-1>=n.

  '''
  if n>2**m-1:
    print('Something is wrong. n<=2^m-1 is not satisfied.')
  b=list()
  for i in range(m):
    b.append(n%2)
    n=n//2
  return np.array(b)

def RM_encoder(m,r,u):
  '''Returns the codeword x corresponding to the message u with respect to the permuted RM(m,r) encoder.
     Here, u is a binom(m,0)+binom(m,1)+...+binom(m,r) array, x is an 2^m array, m,r are integers s.t. 0<=r<=m.

  '''
  n=2**m
  x=np.zeros(n)
  iter=0
  for i in range(n):
    b=num_to_binom(i,m)
    if np.sum(b)<=r:
      summand=u[iter]
      for j in range(m):
        if b[j]==1:
          summand*=(np.arange(n)//(2**j))%2
      x=(x+summand)%2
      iter+=1
  return x

def random_pair_generator(m,r,epsilon):
  '''Returns a random codeword and a generated noisy codeword from the binary symmetric channel with parameter epsilon.
     Here, m,r are integers representing the parameters of the RM(m,r) code, 0<=r<=m, and epsilon is a float s.t. 0<epsilon<0.5.

  '''
  binom=1
  for i in range(1,r+1):
    binom+=math.comb(m,i)
  u=np.random.binomial(1,0.5,binom)
  x=RM_encoder(m,r,u)
  y=(x+np.random.binomial(1,epsilon, 2**m))%2
  return x.copy(), y.copy()

def random_goodpair_generator(m,epsilon):
  '''Returns a random noisy codeword from the binary symmetric channel with parameter epsilon and its maximum likelihood decoding result.
     Here, m is an integer representing the parameter of the RM(m,1) code, 0<=r<=m, and epsilon is a float s.t. 0<epsilon<0.5.

  '''
  u=np.random.binomial(1,0.5,m+1)
  x=RM_encoder(m,1,u)
  y=(x+np.random.binomial(1,epsilon,2**m))%2
  x1=RM_encoder(m,1,FHT_decode(y))
  print(x1.copy(),y.copy())


def FHT(L):
    '''Fast Hadamard transform.
       Here, L is a length 2^m array representing the noisy codeword with 0's substituted by 1 and 1's by -1.
       Returns the Hadamard transformation of vector L.
    '''
    a=L.copy()
    h = 1
    while h < len(a):
      for i in range(0, len(a), h * 2):
        for j in range(i, i + h):
          z = a[j]
          y = a[j + h]
          a[j] = z + y
          a[j + h] = z - y
      a=a/2
      h *= 2
    return a

def FHT_decode(y):
  '''Maximum likelihood decoder of RM(m,1) codes using Fast Hadamard. Outputs the message.
     Here, y is a length 2^m array representing the noisy codeword.

  '''
  m=int(np.log2(np.size(y)))
  a=FHT(1-2*y)
  i=np.argmax(np.abs(a))
  s=np.append(np.array([0]),num_to_binom(i,m))
  if a[i]<0: s[0]=1
  return s

def complementary(m,r):
  n=2**m
  binom=1
  for i in range(1,r+1):
    binom+=math.comb(m,i)
  H=np.zeros((n-binom,n))
  iter=0
  for i in range(n):
    b=num_to_binom(i,m)
    u=np.ones(n)
    if np.sum(b)<=m-r-1:
      for j in range(m):
          if b[j]==1:
            u*=(np.arange(n)//(2**j))%2
      H[iter,:]=u.copy()
      iter+=1
  return H

def accuracy(x, y):
  for i in range(len(x)):
    if int(x[i]) != int(y[i]):
      return 0
  return 1
