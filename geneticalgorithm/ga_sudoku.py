import numpy as np

class ga_sudoku:

 def __init__(self, function, board,pop_ini=None,\
              parameters={'population_size':750,\
                                    'dimension':9,\
                                    'mutation_probability':0.25,\
                                    'elit_ratio': 0.24,\
                                    'crossover_probability': 0.65,\
                                    'parents_portion': 0.4,\
                                    'crossover_type':'uniform'},\
                                     sg=np.random.SeedSequence()):
  if pop_ini is None:                                   
   self.pop_s = parameters['population_size']                                   
  else:
   self.pop_s = pop_ini.shape[0]                                       
  
  self.dim = parameters['dimension']**2
  
  self.par_s=int(parameters['parents_portion']*self.pop_s)
  trl=self.pop_s-self.par_s
  if trl % 2 != 0:
   self.par_s+=1
               
  self.prob_mut=parameters['mutation_probability']
  self.prob_cross=parameters['crossover_probability']
  
  self.c_type=parameters['crossover_type']
  assert (self.c_type=='uniform' or self.c_type=='one_point' or\
          self.c_type=='two_point'),\
  "\n crossover_type must 'uniform', 'one_point', or 'two_point'" 
  
  trl=self.pop_s*parameters['elit_ratio']
  if trl<1 and parameters['elit_ratio'] > 0:
    self.num_elit=1
  else:
    self.num_elit=int(trl)
            
  assert(self.par_s>=self.num_elit), \
  "\n number of parents must be greater than number of elits"
  
  self.rgen = np.random.Generator(np.random.PCG64(sg))
  
  self.board = board
  self.Mask = (self.board == 0).reshape((9,9))
  self.f = function
  self.pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
  
# initial population
  if pop_ini is None:  
   var,obj = None,None       
   for p in range(0,self.pop_s):
     aux = self.board.reshape((9,9)).copy()
     for i in [0,3,6]:
      for j in [0,1,2]:
        i1,i2 = i,i+3
        j1,j2 = 3*j,3*j+3
        tmp = aux[i1:i2,j1:j2]
        msk = self.Mask[i1:i2,j1:j2]
        tmp[msk] = self.rgen.permutation(list({1,2,3,4,5,6,7,8,9} - set(tmp[~msk]))) 
     var = aux.flatten().copy()
     obj = self.f(var)
     self.pop[p]=np.hstack((var,[obj]))
   self.best_variable=var.copy()
   self.best_function=obj
  else:
   self.pop = pop_ini.copy()
   ibest = pop_ini[:,self.dim].argmin()
   self.best_variable=pop_ini[ibest,:self.dim].copy() 
   self.best_function=pop_ini[ibest,self.dim] 
  
 #############################################################
  # Report
  self.report=[]  
   
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
 
 def perm(self,x,msk):
  x[msk] = self.rgen.permutation(x[msk])
  return x
  
 def mut(self,x,m_type):
  ofs = x.reshape((9,9))

  for i in [0,3,6]:
   for j in [0,1,2]:
    i1,i2 = i,i+3
    j1,j2 = 3*j,3*j+3
    aux = ofs[i1:i2,j1:j2].copy()
    msk = self.Mask[i1:i2,j1:j2]
    if self.rgen.random() < self.prob_mut:
     if m_type == 'perm':
       ofs[i1:i2,j1:j2] = self.perm(aux,msk) 
     elif m_type == 'swap':
       aux = aux.flatten()
       k = self.rgen.integers(1,5)
       U,V = self.rgen.integers(1,9,k),self.rgen.integers(1,9,k)
       for u,v in zip(U,V):
        if (u != v) and msk.flatten()[u] and msk.flatten()[v]:
         tmp = aux[u]
         aux[u] = aux[v]
         aux[v] = tmp
       ofs[i1:i2,j1:j2]= aux.reshape(3,3).copy()

  return ofs.flatten() 

 def run(self):
 #############################################################
 #Sort
  self.pop = self.pop[self.pop[:,self.dim].argsort()]              
  if self.pop[0,self.dim] < self.best_function:
    self.best_function=self.pop[0,self.dim].copy()
    self.best_variable=self.pop[0,: self.dim].copy()
  #print(self.pop,'\n')
 #############################################################
 # Report

  self.report.append(self.pop[0,self.dim])
    
 ##############################################################         
 # Normalizing objective function 
            
  normobj=self.pop[:,self.dim].copy()

 #############################################################        
 # Calculate probability
            
  sum_normobj=np.sum(normobj)
  prob=np.zeros(self.pop_s)
  prob=normobj/sum_normobj
  cumprob=np.cumsum(prob)
  
 #############################################################        
 # Select parents
  par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
  for k in range(0,self.num_elit):
   par[k]=self.pop[k].copy()
  
  for k in range(self.num_elit,self.par_s):
   index=np.searchsorted(cumprob,self.rgen.random())
   par[k]=self.pop[index].copy()
              
  ef_par_list=np.array([False]*self.par_s)
  par_count=0
  while par_count==0:
   for k in range(0,self.par_s):
    if self.rgen.random()<=self.prob_cross:
     ef_par_list[k]=True
     par_count+=1
                 
  ef_par=par[ef_par_list].copy()
  
  #############################################################  
  #New generation
  self.pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
  for k in range(0,self.par_s):
   self.pop[k]=par[k].copy()
                
  for k in range(self.par_s, self.pop_s, 2):
   r1=self.rgen.integers(0,par_count)
   r2=self.rgen.integers(0,par_count)
   pvar1=ef_par[r1,: self.dim].copy()
   pvar2=ef_par[r2,: self.dim].copy()
                
   ch=self.cross(pvar1,pvar2,self.c_type)
   ch1=ch[0].copy()
   ch2=ch[1].copy()
            
   ch1=self.mut(ch1,'perm')
   ch2=self.mut(ch2,'swap')
   
   self.pop[k]=np.hstack((ch1,[self.f(ch1)])).copy()
   self.pop[k+1]=np.hstack((ch2,[self.f(ch2)])).copy()               
   
#############################################################
#Sort
  
  self.pop = self.pop[self.pop[:,self.dim].argsort()]
     
  if self.pop[0,self.dim]<self.best_function:
    self.best_function=self.pop[0,self.dim].copy()
    self.best_variable=self.pop[0,: self.dim].copy()
 #############################################################
 # Report
  self.report.append(self.pop[0,self.dim])
  self.output_dict={'variable': self.best_variable, 'function':\
                      self.best_function}
   
