import h5py
import sys
import pandas as pd
hf = h5py.File('sir_valid_rk.hdf5', 'r')
print(len(hf.keys()))
dset = hf['345']
print(dset.shape)
print(dset[:200])