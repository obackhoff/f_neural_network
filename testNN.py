from neural_network import *
import math
import time


def bin2int(binarr, int2bindic, largest):
	for i in range(largest):
		if (int2bindic[i] == np.round(binarr)).all():
			return i
	return -1


int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
sum_largest_number = pow(2, (binary_dim - 1))
mult_largest_number = pow(2, int(binary_dim/2))

binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

X = []
y = []


# for i in range(sum_largest_number):
#     a = i
#     for j in range(sum_largest_number):
#         b = j
#         if np.random.rand() < 0.2:
#             X.append(np.concatenate([int2binary[a],int2binary[b]]))
#             y.append(int2binary[a+b])

# X_p = []
# for i in range(4):
# 	a = np.random.randint(sum_largest_number)
# 	b = np.random.randint(sum_largest_number)
# 	X_p.append(np.concatenate([int2binary[a],int2binary[b]]))
# X_p = np.array(X_p)



for i in range(mult_largest_number):
    a = i
    for j in range(mult_largest_number):
        b = j
        if np.random.rand() < 1:
            X.append(np.concatenate([int2binary[a],int2binary[b]]))
            y.append(int2binary[a*b])

X_p = []
for i in range(8):
	a = np.random.randint(mult_largest_number)
	b = np.random.randint(mult_largest_number)
	X_p.append(np.concatenate([int2binary[a],int2binary[b]]))
X_p = np.array(X_p)

X = np.array(X)
y = np.array(y)

# nn = NeuralNetwork([16,32,8], activation='tanh', output_activation='tanh')
nn = NeuralNetwork([16,32,8], activation='elu', output_activation='sigmoid')

nn.fit(X, y, learning_rate=0.1, epochs=X.shape[0] * 800, all_input=False, lmbda=0)

# nn.save_model('model_mult.nn')

# nn_mult = NeuralNetwork.load_model('model_mult.nn')


print('MULTIPLICATION NN')

for e in X_p:
    p = nn.predict(e)
    print(bin2int(e[0:8], int2binary,largest_number), bin2int(e[8:], int2binary,largest_number), ' -->  ', bin2int(np.abs(p), int2binary,largest_number))



# X_p = np.array([np.concatenate([int2binary[1], int2binary[0]]),
#               np.concatenate([int2binary[5], int2binary[6]]),
#               np.concatenate([int2binary[3], int2binary[11]]),
#               np.concatenate([int2binary[5], int2binary[2]])])



#### TEMP

# print('ADDITION NN')
#
# X_p = []
# for i in range(8):
# 	a = np.random.randint(sum_largest_number)
# 	b = np.random.randint(sum_largest_number)
# 	X_p.append(np.concatenate([int2binary[a],int2binary[b]]))
# X_p = np.array(X_p)
#
# nn_sum = NeuralNetwork.load_model('model_sum.nn')

# nn_sum.fit(X, y, learning_rate=0.2, epochs=X.shape[0] * 20, all_input=False)

# nn_sum.save_model('model_sum.nn');

# for e in X_p:
#     p = nn_sum.predict(e)
#     print(bin2int(e[0:8], int2binary,largest_number), bin2int(e[8:], int2binary,largest_number), ' -->  ', bin2int(np.abs(p), int2binary,largest_number))
