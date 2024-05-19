import numpy as np
import neurolab as nl

# Визначення піксельних матриць для літер Д, Н, Г
D = [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1]

N = [1, 0, 0, 0, 1,
     1, 1, 0, 0, 1,
     1, 0, 1, 0, 1,
     1, 0, 0, 1, 1,
     1, 0, 0, 0, 1]

G = [0, 1, 1, 1, 0,
     1, 0, 0, 0, 0,
     1, 0, 1, 1, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0]

chars = ['D', 'N', 'G']
target = [D, N, G]
target = np.asfarray(target)
target[target == 0] = -1

# Створення та навчання мережі Хопфілда
net = nl.net.newhop(target)

# Тестування на тренувальних зразках
output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

# Внесення незначних помилок у тестові зразки та перевірка розпізнавання

print("\nTest on defaced D:")
test_D = [1, 1, 1, 1, 1,
          1, 0, 0, 0, 1,
          1, 0, 0, 0, 1,
          1, 0, 0, 1, 1,
          1, 1, 1, 1, 0] # Змінений останній піксель
test_D = np.asfarray(test_D)
test_D[test_D == 0] = -1
out_D = net.sim([test_D])
print((out_D[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))

print("\nTest on defaced N:")
test_N = [1, 0, 0, 0, 1,
          1, 1, 0, 0, 1,
          1, 0, 1, 0, 1,
          1, 0, 0, 1, 0, # Змінений останній піксель
          1, 0, 0, 0, 1]
test_N = np.asfarray(test_N)
test_N[test_N == 0] = -1
out_N = net.sim([test_N])
print((out_N[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))

print("\nTest on defaced G:")
test_G = [0, 1, 1, 1, 0,
          1, 0, 0, 0, 0,
          1, 0, 1, 1, 0, # Змінений останній піксель
          1, 0, 0, 0, 1,
          0, 1, 1, 1, 0]
test_G = np.asfarray(test_G)
test_G[test_G == 0] = -1
out_G = net.sim([test_G])
print((out_G[0] == target[2]).all(), 'Sim. steps', len(net.layers[0].outs))
