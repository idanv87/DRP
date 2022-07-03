

import numpy as np
import pickle
import matplotlib.pyplot as plt
import utils
import matplotlib.pyplot as plt
import torch
import math
import matplotlib.animation as animation

from constants import Constants
with open('files/e_test.pkl', 'rb') as file:
    e_true = pickle.load(file)
with open('files/hx_test.pkl', 'rb') as file:
    hx_true = pickle.load(file)
with open('files/hy_test.pkl', 'rb') as file:
    hy_true = pickle.load(file)
# Grid parameters.
k1, k2 = Constants.K1_TEST[0], Constants.K2_TEST[0]




E = np.squeeze(e_true[0,:,:,:]).copy()
Hx = np.squeeze(hx_true[0,:,:,:]).copy()
Hy = np.squeeze(hy_true[0,:,:,:]).copy()


Z = 1
error=0;
for n in range(Constants.TIME_STEPS-1):
    E[1:Constants.N - 1, 1:Constants.N - 1] = E[1:Constants.N - 1, 1:Constants.N - 1] + \
        (Constants.DT / Constants.DX) * np.diff(Hy,axis=0) - (Constants.DT / Constants.DY) * np.diff(Hx,axis=1)
    Hx -=  (Constants.DT / ( Constants.DY)) * (np.diff(E[1:-1,:],axis=1))
    Hy +=  (Constants.DT / ( Constants.DX)) * (np.diff(E[:,1:-1],axis=0))
    #error+=np.abs(E-np.squeeze(e_true[n+1,:,:,])).max()+np.abs(Hx-np.squeeze(hx_true[n+1,:,:,])).max()+np.abs(Hy-np.squeeze(hy_true[n+1,:,:,])).max()

