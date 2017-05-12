from neural_network import *
import math
import time


# nn = NeuralNetwork([2,8,16,1], activation='tanh', output_activation='lin')
nn = NeuralNetwork([2,16,16,1], activation='sigmoid', output_activation='lin')

X_1 = []
y_1 = []

for i in range(300):
    a = np.random.random() * 10 * (lambda x: -1 if x < 0.5 else 1)(np.random.rand())
    b = np.random.random() * 10 * (lambda x: -1 if x < 0.5 else 1)(np.random.rand())
    X_1.append(np.array([a,b]))
    y_1.append(a*b)


X_1 = np.array(X_1)
y_1 = np.array(y_1)

nn.fit(X_1, y_1, learning_rate=0.0001, epochs=X_1.shape[0] * 2000, all_input=False)

X_p = np.array([[0.5, 0.1],
              [5, 5],
              [15, 3],
              [3, -2],
              [8, 6],
              [12, 20]])

avg_err = 0
for e in X_p:
    p = nn.predict(e)
    avg_err += np.abs((e[0] * e[1]) - (p)[0])
    print(e,' -->  ', (p)[0])
avg_err /= X_p.shape[0]
print('avg_err:', avg_err)
