from neural_network import NeuralNetwork
import numpy as np


# nn = NeuralNetwork([2,8,16,1], activation='tanh', output_activation='lin')

X = []
y = []

# for i in range(3000):
#     a = np.random.random() * 15 * (lambda x: -1 if x < 0.5 else 1)(np.random.rand())
#     b = np.random.random() * 15 * (lambda x: -1 if x < 0.5 else 1)(np.random.rand())
#     X.append(np.array([a,b]))
#     y.append(a*b)


X_p = [[1, 3],
      [2, 8],
      [3, 11],
      [4, 2],
      [5, 3],
      [6, 12],
      [7, 9],
      [8, 10],
      [14, 9],
      [11, 2],
      [0.5, 0.1],
      [5, 5],
      [15, 3],
      [3, -2],
      [8, 6],
      [12, 20]]


X_p = np.array(X_p)

y_raw = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [2], [4], [5], [6], [7], [9], [10], [12], [13], [14], [15], [0], [2], [4], [6], [8], [10], [12], [14], [18], [20], [26], [28], [30], [0], [3], [6], [9], [12], [15], [18], [21], [27], [30], [36], [39], [42], [45], [0], [4], [12], [16], [20], [24], [28], [32], [36], [40], [44], [48], [52], [56], [60], [0], [10], [25], [35], [45], [50], [60], [65], [75], [0], [6], [12], [18], [30], [36], [42], [48], [54], [60], [66], [78], [84], [0], [7], [21], [28], [35], [42], [56], [70], [77], [84], [91], [98], [0], [8], [32], [40], [48], [56], [64], [72], [88], [96], [104], [9], [18], [36], [54], [63], [72], [81], [99], [108], [117], [135], [0], [10], [20], [40], [60], [70], [90], [100], [110], [120], [140], [33], [55], [66], [77], [88], [99], [132], [143], [154], [165], [0], [24], [36], [48], [60], [72], [84], [96], [108], [120], [132], [156], [168], [180], [0], [13], [26], [39], [52], [78], [91], [104], [130], [143], [156], [169], [182], [195], [14], [28], [42], [70], [84], [98], [112], [140], [154], [168], [182], [196], [210], [15], [30], [45], [60], [75], [90], [105], [120], [150], [165], [195], [210], [225]]
X_raw = [[0, 3], [0, 4], [0, 5], [0, 7], [0, 8], [0, 10], [0, 11], [0, 14], [0, 15], [1, 0], [1, 1], [1, 2], [1, 4], [1, 5], [1, 6], [1, 7], [1, 9], [1, 10], [1, 12], [1, 13], [1, 14], [1, 15], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 9], [2, 10], [2, 13], [2, 14], [2, 15], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 9], [3, 10], [3, 12], [3, 13], [3, 14], [3, 15], [4, 0], [4, 1], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [5, 0], [5, 2], [5, 5], [5, 7], [5, 9], [5, 10], [5, 12], [5, 13], [5, 15], [6, 0], [6, 1], [6, 2], [6, 3], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 13], [6, 14], [7, 0], [7, 1], [7, 3], [7, 4], [7, 5], [7, 6], [7, 8], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [8, 0], [8, 1], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 11], [8, 12], [8, 13], [9, 1], [9, 2], [9, 4], [9, 6], [9, 7], [9, 8], [9, 9], [9, 11], [9, 12], [9, 13], [9, 15], [10, 0], [10, 1], [10, 2], [10, 4], [10, 6], [10, 7], [10, 9], [10, 10], [10, 11], [10, 12], [10, 14], [11, 3], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 12], [11, 13], [11, 14], [11, 15], [12, 0], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 13], [12, 14], [12, 15], [13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 6], [13, 7], [13, 8], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [14, 1], [14, 2], [14, 3], [14, 5], [14, 6], [14, 7], [14, 8], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 10], [15, 11], [15, 13], [15, 14], [15, 15]]
for i in range(len(X_raw)):
    x1 = X_raw[i][0]
    x2 = X_raw[i][1]
    X.append([x1, x2])
    y.append(y_raw[i][0])

X = np.array(X)
y = np.array(y)

nn = NeuralNetwork([2,8,8,1], activation='elu', output_activation='lin')
nn.fit(X, y, learning_rate=0.0001, epochs=X.shape[0] * 2000, all_input=False, lmbda=0.0001)




tot_err = 0
for e in X_p:
    p = nn.predict(e)
    tot_err += np.abs((e[0] * e[1]) - (p)[0])
    print(e,' -->  ', (p)[0])
# avg_err /= X_p.shape[0]
print('tot_err:', tot_err)