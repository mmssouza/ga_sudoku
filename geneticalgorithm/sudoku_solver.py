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
 seedseq = p[2]

 model=ga(function=f,board=board,algorithm_parameters = params,progress_bar = False,sg = seedseq)
 model.run()
 return (p[1],model.best_variable,model.report)

N = 20 

ss = np.random.SeedSequence(345789)

params = {'max_num_iteration': 500,\
                   'population_size': 1000,\
                   'mutation_probability': 0.25,\
                   'elit_ratio': 0.5,\
                   'crossover_probability': 0.6,\
                   'parents_portion': 0.6,\
                   'crossover_type':'one_point',\
                   'max_iteration_without_improv':None}

if __name__ == '__main__':
 for k in d:
  board = d[k].flatten()
  print(k)
  with Pool(processes=4) as pool:
   res = pool.map(task,[(board,params,s) for s in ss.spawn(N)])
  print("*******************************************************\n")

#  for i in res:
#   print('\n',i[0])
#   print('\n',i[1])
#   print('\n',i[2])
#   print()
