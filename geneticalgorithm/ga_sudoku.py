import numpy as np

from geneticalgorithm import geneticalgorithm as ga

class ga_sudoku(ga):

 def __init__(self, function, board,\
                 algorithm_parameters={'max_num_iteration': 500,\
                                       'population_size':650,\
                                       'mutation_probability':0.25,\
                                       'elit_ratio': 0.24,\
                                       'crossover_probability': 0.75,\
                                       'parents_portion': 0.4,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None},\
                     convergence_curve=False,\
                         progress_bar=True,\
                         sg=np.random.SeedSequence()):

  self.rgen = np.random.Generator(np.random.PCG64(sg))

  self.board = board

  super().__init__(function,dimension = 81,variable_type = 'int',variable_boundaries = np.array([[1,9]]*81,dtype='int'),variable_type_mixed = None,function_timeout = 10,algorithm_parameters=algorithm_parameters,convergence_curve=convergence_curve, progress_bar=progress_bar)

 def perm(self,x,msk):
  x[~msk] = self.rgen.permutation(x[~msk])
  return x

 def init(self):
        ############################################################# 
        # Initial Population
        
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        
        self.pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        self.solo=np.zeros(self.dim+1)
        self.var=np.zeros(self.dim)       
        
        self.Mask = (self.board != 0).reshape((9,9))

        for p in range(0,self.pop_s):
         aux = self.board.reshape((9,9)).copy()

         for i in [0,3,6]:
          for j in [0,1,2]:
           i1,i2 = i,i+3
           j1,j2 = 3*j,3*j+3
           tmp = aux[i1:i2,j1:j2]
           msk = self.Mask[i1:i2,j1:j2]
           tmp[~msk] = self.rgen.permutation(list({1,2,3,4,5,6,7,8,9} - set(tmp[msk]))) 

         self.var = aux.flatten().copy()
         self.solo[0:self.dim] = self.var.copy()
         self.obj=self.sim(self.var)            
         self.solo[self.dim]=self.obj
         self.pop[p]=self.solo.copy()
   
 def cross(self,x,y,c_type):

  X = x.reshape((9,9))
  Y = y.reshape((9,9))
  ofs1 = X.copy()
  ofs2 = Y.copy()
  if c_type == 'uniform':
   for i in [0,3,6]:
    for j in [0,1,2]:
     i1,i2 = i,i+3
     j1,j2 = 3*j,3*j+3
     ran=self.rgen.random()
     if ran < 0.5:
      ofs1[i1:i2,j1:j2] = Y[i1:i2,j1:j2].copy()
      ofs2[i1:i2,j1:j2] = X[i1:i2,j1:j2].copy() 
  elif c_type == 'one_point':
    p = self.rgen.integers(0,8)
    k = 0
    for i in [0,3,6]:
     for j in [0,1,2]:
      i1,i2 = i,i+3
      j1,j2 = 3*j,3*j+3
      if k <= p:
       ofs1[i1:i2,j1:j2] = Y[i1:i2,j1:j2].copy()
       ofs2[i1:i2,j1:j2] = X[i1:i2,j1:j2].copy()
      k = k + 1 
  elif c_type == 'two_point':
    k = 0
    p1 = self.rgen.integers(0,9)
    p2 = self.rgen.integers(p1,9)
    for i in [0,3,6]:
     for j in [0,1,2]:
      i1,i2 = i,i+3
      j1,j2 = 3*j,3*j+3
      if p1 <= k <= p2:
       ofs1[i1:i2,j1:j2] = Y[i1:i2,j1:j2].copy()
       ofs2[i1:i2,j1:j2] = X[i1:i2,j1:j2].copy()
      k = k + 1 

  return([ofs1.flatten(), ofs2.flatten()])          

 def mut(self,x,m_type):

  ofs = x.reshape((9,9))

  for i in [0,3,6]:
   for j in [0,1,2]:
    i1,i2 = i,i+3
    j1,j2 = 3*j,3*j+3
    aux = ofs[i1:i2,j1:j2].copy()
    msk = self.Mask[i1:i2,j1:j2]
    if m_type == 'perm':
     if self.rgen.random() < self.prob_mut:
      ofs[i1:i2,j1:j2] = self.perm(aux,msk) 
    elif m_type == 'swap':
     aux = aux.flatten()
     for k in range(9):
      if self.rgen.random() < self.prob_mut:
       l = self.rgen.integers(9)
       if (k != l) and ~(msk.flatten()[k] or msk.flatten()[l]):
        tmp = aux[k]
        aux[k] = aux[l]
        aux[l] = tmp
     ofs[i1:i2,j1:j2]= aux.reshape(3,3).copy() 

  return ofs.flatten() 


