
import numpy as np
import itertools
import math



class de:

 self.perm_tbl = np.array([p for p in itertools.permutation([1,2,3,4,5,6,7,8,9])])
 self.N_perm = self.perm_tbl.shape[0]
 self.float2index = lambda flt: math.trunc(y*(N_perm - 1))
 self.index2float = lambda idx : idx/(N_perm - 1)
 
 def __init__(self,fitness_func,board,npop = 10,pr = 0.7,beta = 2.5,sg = np.random.SeedSequence()):

  self.rng = np.random.Generator(np.random.PCG64(sg))
  self.board = board
  self.ns = int(npop)
  self.beta = beta
  self.pr  = pr 
  self.ff = fitness_func
  self.pop = np.array([self.gera_individuo() for i in range(self.ns)])
  self.fit = np.array([self.ff(i) for i in self.pop])

 def gera_individuo(self):

   return 1. - 1.*self.rng.random(9) 
    
 def run(self):  
   
  for i in scipy.arange(self.ns):
   # para cada individuo da populacao 
   # gera trial vector usado para perturbar individuo atual (indice i)
   # a partir de 3 individuos escolhidos aleatoriamente na populacao e
   # cujos indices sejam distintos e diferentes de i
   invalido = True
   while invalido:
    j = self.rng.integers(0,self.ns-1,3)
    invalido = (i in j)
    invalido = invalido or (j[0] == j[1]) 
    invalido = invalido or (j[1] == j[2]) 
    invalido = invalido or (j[2] == j[0])
   
   # trial vector a partir da mutacao de um alvo 
   u = self.pop[j[0]] + self.beta*(self.pop[j[1]] - self.pop[j[2]])
 
   # gera por crossover solucao candidata
   c = self.pop[i].copy()  
   # seleciona indices para crossover
   # garantindo que ocorra crossover em
   # pelo menos uma vez
   j = self.rng.integers(0,self.pop.shape[1]-1)
  
   for k in np.arange(self.pop.shape[1]):
    if (self.rng.random() < self.pr) or (k == j):
     c[k] = u[k]  

   c_fit = self.ff(c) 
   self.fit[i] = self.ff(self.pop[i])
       
   # leva para proxima geracao quem tiver melhor fitness
   if c_fit < self.fit[i]:
    self.pop[i] = c
    self.fit[i] = c_fit
 
