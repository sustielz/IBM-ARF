import numpy as np 
import sys
import os

DATANAME = 'data/run{}.npz'
data = np.load(DATANAME.format(0))
data = dict(data)
NRUN = 1 
while os.path.exists(DATANAME.format(NRUN)):
    next_data = np.load(DATANAME.format(NRUN))
    for key in list(data.keys()): data[key] = np.append(data[key], next_data[key], axis=0)
    NRUN += 1

with open('datac/run.npz'.format(FILENAME), 'wb') as f:
    np.savez(f, **data)

for key in list(data.keys()): 
    try:
        data[key]=data[key][-1]
    except:
        pass
with open('dataf/run.npz'.format(FILENAME), 'wb') as f:
    np.savez(f, **data)
