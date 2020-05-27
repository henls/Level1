import time, os
import cupy as cp
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Pool
def a(m,n):
    ma = np.arange(m*n).reshape(m,n)
    a = cp.array(ma)
    st = time.time()
    print(cp.fft.fft(a))
    print(time.time()-st)
#a1 = multiprocessing.Process(target=a,args=(2,3))
#a1.start()
#a2 = multiprocessing.Process(target=a,args=(2,3))
#a2.start()
#a3 = multiprocessing.Process(target=a,args=(2,3))
#a3.start()
mp.set_start_method('spawn')
manager = mp.Manager()
q = manager.Queue()
p = Pool()    # create producer
# start 10 processor for consumer    # each consumer will run algorithm with model on GPU, we load model for each process    
for i in range(3):
    p.apply_async(a, args=(2,3))
p.close()
p.join()

