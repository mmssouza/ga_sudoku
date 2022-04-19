#!/usr/bin/python3 -u

import numpy as np
from ga_sudoku import ga_sudoku as ga
from multiprocessing import Pool
import pickle

#board = np.array(   [0, 0, 4, 5, 1, 0, 7, 0, 8, 
#                     0, 0, 0, 0, 0, 0, 9, 0, 0, 
#                     8, 0, 9, 6, 2, 0, 0, 3, 0, 
#                     0, 2, 8, 3, 6, 4, 1, 0, 0, 
#                     0, 0, 3, 0, 0, 0, 2, 0, 0, 
#                     0, 0, 1, 9, 8, 2, 6, 5, 0, 
#                     0, 8, 0, 0, 7, 6, 3, 0, 9, 
#                     0, 0, 6, 0, 0, 0, 0, 0, 0, 
#                     4, 0, 7, 0, 3, 5, 8, 0, 0],dtype='int')
                    
#board = np.array(  [0, 4, 2, 0, 0, 0, 0, 0, 5, 
#                    0, 0, 0, 6, 3, 2, 0, 8, 0, 
#                    0, 8, 0, 0, 4, 0, 2, 0, 0, 
#                    0, 0, 0, 0, 0, 0, 0, 0, 0, 
#                    7, 1, 5, 0, 6, 8, 3, 4, 0, 
#                    9, 0, 8, 3, 5, 0, 7, 6, 1, 
#                    0, 9, 1, 0, 0, 6, 0, 0, 0, 
#                    0, 0, 0, 0, 2, 0, 1, 9, 0, 
#                    0, 0, 6, 1, 0, 0, 0, 5, 0],dtype='int')                      

fname ='v2_train.pkl'

d = pickle.load(open(fname,'rb'))

def f(X):

 x = X.reshape((9,9))
 g = lambda x: (np.array([(x == l).sum() for l in range(1,10)]) > 1).sum()
 sum_row = np.zeros(9,dtype='int')
 sum_col = sum_row.copy()

 for i in range(9):
  sum_row[i] = g(x[i])
  sum_col[i] = g(x[:,i])

 return sum_row.sum() + sum_col.sum() 

def task(p):
 board = p[0]
 params = p[1]
 N_iter = params['max_num_iteration']
 seedseq = p[2]

 model=ga(function=f,board=board,parameters = params,sg = seedseq)
 [model.run() for _ in range(N_iter)]
 return model.pop

N = 4 

ss = np.random.SeedSequence(345789)

params = {'max_num_iteration': 450,\
                   'population_size': 250,\
                   'dimension': 9,\
                   'mutation_probability': 0.15,\
                   'elit_ratio': 0.25,\
                   'crossover_probability': 0.75,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


params2 = {'max_num_iteration': 50,\
                   'population_size': 100,\
                   'dimension': 9,\
                   'mutation_probability': 0.25,\
                   'elit_ratio': 0.4,\
                   'crossover_probability': 0.6,\
                   'parents_portion': 0.5,\
                   'crossover_type':'two_point',\
                   'max_iteration_without_improv':None}

if __name__ == '__main__':
 n = 12
 for k in d:
  print(k)
  if n == 0:
   break;
  n-=1;
  board = d[k].flatten()
  with Pool(processes=4) as pool:
   res = pool.map(task,[(board,params,s) for s in ss.spawn(N)])
  print("*******************************************************\n")
  p_ini = np.vstack([r[:100,:] for r in res])
  print(p_ini.shape)
  model = ga(function=f, board=board,parameters = params2,pop_ini=p_ini)
  for i in range(150):
   model.run()
   print(model.pop[0,9*9],end=' ')
  print()
