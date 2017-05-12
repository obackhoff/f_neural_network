from neural_network import *
import math
import time

nn = NeuralNetwork([1,8,16,1], activation='elu', output_activation='lin')
# nn = NeuralNetwork([1,4,4,1], activation='sigmoid', output_activation='lin')

X_1 = []
y_1 = []

for i in range(10000):
    a = np.random.random() * 10 * (lambda x: -1 if x < 0.5 else 1)(np.random.rand())
    X_1.append(np.array([a]))
    y_1.append(a*a)


X_1 = np.array(X_1)
y_1 = np.array(y_1)

nn.fit(X_1, y_1, learning_rate=0.00001, epochs=X_1.shape[0] * 100, all_input=False)

X_p = np.array([[0.5],
              [-2],
              [1],
              [0],
              [3],
              [12]])

avg_err = 0
for e in X_p:
    p = nn.predict(e)
    avg_err += np.abs((e[0] * e[0]) - (p)[0])
    print(e,' -->  ', (p)[0])
avg_err /= X_p.shape[0]
print('avg_err:', avg_err)
