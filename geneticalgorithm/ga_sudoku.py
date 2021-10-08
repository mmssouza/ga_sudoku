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
                         progress_bar=True):


  self.board = board
  super().__init__(function,dimension = 81,variable_type = 'int',variable_boundaries = np.array([[1,9]]*81,dtype='int'),variable_type_mixed = None,function_timeout = 10,algorithm_parameters=algorithm_parameters,convergence_curve=convergence_curve, progress_bar=progress_bar)

 def perm(self,x,msk):
  x[~msk] = np.random.permutation(x[~msk])
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
           tmp[~msk] = np.random.permutation(list({1,2,3,4,5,6,7,8,9} - set(tmp[msk]))) 

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
  if self.c_type == 'uniform':
   for i in [0,3,6]:
    for j in [0,1,2]:
     i1,i2 = i,i+3
     j1,j2 = 3*j,3*j+3
     ran=np.random.random()
     if ran < 0.5:
      ofs1[i1:i2,j1:j2] = Y[i1:i2,j1:j2].copy()
      ofs2[i1:i2,j1:j2] = X[i1:i2,j1:j2].copy() 
  elif self.c_type == 'one_point':
    p = np.random.randint(0,8)
    k = 0
    for i in [0,3,6]:
     for j in [0,1,2]:
      i1,i2 = i,i+3
      j1,j2 = 3*j,3*j+3
      if k <= p:
       ofs1[i1:i2,j1:j2] = Y[i1:i2,j1:j2].copy()
       ofs2[i1:i2,j1:j2] = X[i1:i2,j1:j2].copy()
      k = k + 1 
  elif self.c_type == 'two_point':
    k = 0
    p1 = np.random.randint(0,9)
    p2 = np.random.randint(p1,9)
    for i in [0,3,6]:
     for j in [0,1,2]:
      i1,i2 = i,i+3
      j1,j2 = 3*j,3*j+3
      if p1 <= k <= p2:
       ofs1[i1:i2,j1:j2] = Y[i1:i2,j1:j2].copy()
       ofs2[i1:i2,j1:j2] = X[i1:i2,j1:j2].copy()
      k = k + 1 

  return([ofs1.flatten(), ofs2.flatten()])          

 def mut(self,x):

  ofs = x.reshape((9,9))

  for i in [0,3,6]:
   for j in [0,1,2]:
    i1,i2 = i,i+3
    j1,j2 = 3*j,3*j+3
    ran=np.random.random()
    if ran < self.prob_mut:
     aux = ofs[i1:i2,j1:j2].copy()
     msk = self.Mask[i1:i2,j1:j2]
     ofs[i1:i2,j1:j2] = self.perm(aux,msk) 

  return ofs.flatten() 


