from fnn import run
import sys
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)

def func(x):
    return 0.2*x**4 - x**2

x = 5 * np.random.random(20) - 2.5
y = func(x)

t = np.dstack((x,y))[0]
t_str = np.array2string(t, separator=',').replace('\n', '')

p = np.reshape(np.arange(-4,4,0.1),(-1,1))
p_str = np.array2string(p, separator=',').replace('\n', '')

sys.argv = ['prog','-Q', '-a', 'elu','-o', 'lin','-n', '[1,4,4,1]', '-t', t_str, '-p', p_str, '-e', 50000, '-l', 0.001]
preds = run()

plt.plot(p,preds, color='red')
plt.scatter(x,y)
