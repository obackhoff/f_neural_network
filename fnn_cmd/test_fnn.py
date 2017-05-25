#import matplotlib
#matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
from fnn import run
import sys
import matplotlib.pyplot as plt
import numpy as np

DATA_POINTS = 50
NOISE_LEVEL = 1
FUNC = 4

np.set_printoptions(threshold=np.nan)

def func(x):
    if(FUNC == 0):
        return 5*np.sin(x) + 5*np.cos(2*x)
    elif (FUNC == 1):
        return 0.1*x**4 - 2*x**2 - 2 
    elif (FUNC == 2):
        return 0.5*x + 2 
    elif (FUNC == 3):
        return 5*np.exp(-0.5*x**2)
    elif (FUNC == 4):
        return 5*x/(np.abs(x)+1.15)

x = 10 * np.random.random(DATA_POINTS) - 5
y = func(x)
y = y + (NOISE_LEVEL*np.random.random(DATA_POINTS) - NOISE_LEVEL/2)

t = np.dstack((x,y))[0]
t_str = np.array2string(t, separator=',').replace('\n', '')

p = np.reshape(np.arange(-5,5,0.1),(-1,1))
p_str = np.array2string(p, separator=',').replace('\n', '')

sys.argv = ['prog','-Q', '-a', 'elu','-o', 'lin','-n', '[1,8,8,1]', '-t', t_str, '-p', p_str, '-e', 40000,'-r', 5000 , '-l', 0.001, '-L', 0.01]
preds = run()

plt.plot(p,preds, color='red', label='Prediction')
plt.plot(p, func(p), color='green', label='Truth', alpha=0.5)
plt.scatter(x, y, label='Training set')
plt.legend()
plt.show()