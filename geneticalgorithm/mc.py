#!/usr/bin/python3 -u
import numpy as np
from ga_sudoku import ga_sudoku as ga
from multiprocessing import Pool

board = np.array(  [0, 4, 2, 0, 0, 0, 0, 0, 5, 
                    0, 0, 0, 6, 3, 2, 0, 8, 0, 
                    0, 8, 0, 0, 4, 0, 2, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    7, 1, 5, 0, 6, 8, 3, 4, 0, 
                    9, 0, 8, 3, 5, 0, 7, 6, 1, 
                    0, 9, 1, 0, 0, 6, 0, 0, 0, 
                    0, 0, 0, 0, 2, 0, 1, 9, 0, 
                    0, 0, 6, 1, 0, 0, 0, 5, 0],dtype='int')                      

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
 model=ga(function=f,board=board,algorithm_parameters = p,progress_bar = False)
 model.run()
 return (p,model.best_variable,model.report)

p_cross_l = np.linspace(0.35,0.95,6) 
p_mut_l = np.linspace(0.05,0.55,6)
par_l = [0.1,0.2,0.4,0.5,0.6,0.7] 
elit_l = [0.2,0.4,0.5,0.7,0.8,0.9]
npop_l = [150,250,500,1000,1500,2000]
ctype_l = 2*['uniform','one_point','two_point']

pl = []
for i in range(50):
 ran = np.random.randint(0,6,6)
 p = {'max_num_iteration': 10000,\
                    'population_size': npop_l[ran[0]],\
                    'mutation_probability': p_mut_l[ran[1]],\
                    'elit_ratio': elit_l[ran[2]]*par_l[ran[3]],\
                    'crossover_probability': p_cross_l[ran[4]],\
                    'parents_portion': par_l[ran[3]],\
                   'crossover_type':ctype_l[ran[5]],\
                    'max_iteration_without_improv':2500}
 
 pl.append(p) 

if __name__ == '__main__':
 with Pool(processes=6) as pool:
  res = pool.map(task,pl)
  for i in res:
   print("\n********************************")
   print('\n',i[0])
   print('\n',i[1])
   print('\n',i[2])

