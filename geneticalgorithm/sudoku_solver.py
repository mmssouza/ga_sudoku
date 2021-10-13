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

board_name ='image1000.dat'
dataset_fname ='v2_train.pkl'

d = pickle.load(open(dataset_fname,'rb'))

board = d[board_name].flatten()

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
 model=ga(function=f,board=board,algorithm_parameters = p[0],progress_bar = False,sg = p[1])
 model.run()
 return (p[0],model.best_variable,model.report)

N = 30

ss = np.random.SeedSequence(12345)

params = {'max_num_iteration': 5700,\
                   'population_size': 1350,\
                   'mutation_probability': 0.26,\
                   'elit_ratio': 0.3,\
                   'crossover_probability': 0.8,\
                   'parents_portion': 0.4,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':2500}

if __name__ == '__main__':
 with Pool(processes=4) as pool:
  res = pool.map(task,[(params,s) for s in ss.spawn(N)])
  for i in res:
   print("\n********************************")
   print('\n',i[0])
   print('\n',i[1])
   print('\n',i[2])
   print()
