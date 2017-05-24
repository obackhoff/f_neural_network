from fnn import run
import sys
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)

def func(x):
    return 0.1*x**4 - 2*x**2 - 2

x = 10 * np.random.random(20) - 5
y = func(x)

t = np.dstack((x,y))[0]
t_str = np.array2string(t, separator=',').replace('\n', '')

p = np.reshape(np.arange(-5,5,0.1),(-1,1))
p_str = np.array2string(p, separator=',').replace('\n', '')

sys.argv = ['prog','-Q', '-a', 'elu','-o', 'lin','-n', '[1,8,8,1]', '-t', t_str, '-p', p_str, '-e', 50000, '-l', 0.001, '-L', 0.1]
preds = run()

plt.plot(p,preds, color='red', label='Prediction')
plt.plot(p, func(p), color='green', label='Truth', alpha=0.5)
plt.scatter(x, y, label='Training set')
plt.legend()
plt.show()