import os
import sys
import pickle as pk
import numpy as np
import pandas as pd
import random as rn

import matplotlib.pyplot as plt

plt.ion()

d_25_100 = pk.load(open('residuals_epochs25_batchSize100.pkl','rb'))
d_25_10 = pk.load(open('residuals_epochs25_batchSize10.pkl','rb'))
d_25_1 = pk.load(open('residuals_epochs25_batchSize1.pkl','rb'))
d_100_1 = pk.load(open('residuals_epochs100_batchSize1.pkl','rb'))


#plt.figure(1)
#plt.clf()
#plt.plot(d_25_100['prediction'][:,1], d_25_100['prediction'][:,0], 'o', color='r', alpha=0.1, label='Model 0')
#plt.plot(d_25_10['prediction'][:,1], d_25_10['prediction'][:,0], 'o', color='b', alpha=0.1, label='Model 1')
#plt.plot(d_25_1['prediction'][:,1], d_25_1['prediction'][:,0], 'o', color='cyan', alpha=0.1, label='Model 2')
#plt.plot(d_25_1['expected_val'][:,1], d_25_1['expected_val'][:,0], 'o', alpha=0.1, color='k', label='Expected')
#plt.title('Pointing Offset Predictions on Validation Data',fontsize=20)
#plt.legend(loc='best')
#plt.xlabel('Azimuth [arcsec]', fontsize=20)
#plt.ylabel('Elevation [arcsec]', fontsize=20)


plt.figure(1)
plt.clf()
plt.plot(d_25_100['residuals'][:,1], d_25_100['residuals'][:,0], 'o', color='r', alpha=0.2, label='Model 0')
plt.plot(d_25_10['residuals'][:,1], d_25_10['residuals'][:,0], 'o', color='b', alpha=0.2, label='Model 1')
plt.plot(d_25_1['residuals'][:,1], d_25_1['residuals'][:,0], 'o', color='c', alpha=0.2, label='Model 2')
#plt.plot(d_100_1['residuals'][:,1], d_100_1['residuals'][:,0], 'o', color='m', alpha=0.2, label='Model 3')
plt.title('Pointing Residuals on Validation Data',fontsize=15)
plt.legend(loc='best')
plt.xlabel('Azimuth [arcsec]', fontsize=15)
plt.ylabel('Elevation [arcsec]', fontsize=15)
plt.savefig('residuals.png')
