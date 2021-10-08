import numpy

board = numpy.array([0, 0, 4, 5, 1, 0, 7, 0, 8, 
                     0, 0, 0, 0, 0, 0, 9, 0, 0, 
                     8, 0, 9, 6, 2, 0, 0, 3, 0, 
                     0, 2, 8, 3, 6, 4, 1, 0, 0, 
                     0, 0, 3, 0, 0, 0, 2, 0, 0, 
                     0, 0, 1, 9, 8, 2, 6, 5, 0, 
                     0, 8, 0, 0, 7, 6, 3, 0, 9, 
                     0, 0, 6, 0, 0, 0, 0, 0, 0, 
                     4, 0, 7, 0, 3, 5, 8, 0, 0],dtype='int')
 
a = numpy.array([4., 7. ,4. ,5. ,1. ,7. ,7. ,7. ,9. ,2. ,6. ,1. ,2. ,4. ,2. ,7. ,6. ,6. ,9. ,1. ,9. ,6. ,2. ,9. ,9. ,3. ,9. ,2. ,2. ,8. ,3. ,6. ,4. ,1. ,4. ,2. ,6. ,4. ,3. ,5. ,5. ,1. ,2. ,8. ,5. ,6. ,9. ,2. ,9. ,8. ,3. ,6. ,5. ,9. ,7. ,8. ,6. ,4. ,6. ,6. ,3. ,9. ,9. ,1. ,3. ,6. ,5. ,2. ,8. ,7. ,2. ,9. ,5. ,5. ,7. ,5. ,3. ,5. ,8. ,1. ,1.])

def pen(X):
 ix = numpy.where(board != 0)[0]
 return (((X[ix] - board[ix])/board[ix])**2).sum() 
 
def f1(X):
 x = X.reshape(9,9)
 sum_c = x.sum(axis=0) - 45
 sum_l = x.sum(axis=1) - 45
 sum_b = []
 for i in [(0,3),(3,6),(6,9)]:
  for j in [(0,3),(3,6),(6,9)]:
   sum_b.append(x[i[0]:i[1],j[0]:j[1]].sum() - 45)
 sum_b = numpy.array(sum_b)
 ix = numpy.where(board != 0)[0]
 return (numpy.hstack((sum_c, sum_l, sum_b))**2).sum() 

def f2(X):
 x = X.reshape((9,9))
 sum_l = numpy.zeros(9,dtype='int')
 sum_c = numpy.zeros(9,dtype='int')
 for i in range(9):
  aux = numpy.array([(x[i] == j).sum() for j in range(1,10)])
  sum_l[i] = (aux > 1).sum()
  aux = numpy.array([(x[:,i] == j).sum() for j in range(1,10)])
  sum_c[i] = (aux > 1).sum()
 return sum_c.sum() + sum_l.sum() 


