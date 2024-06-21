import numpy as np

arr = np.array([np.array([1, 2]), np.array([11, 22]), np.array([1.1, 2.2])])

x = np.array([1.1, 2.2])
w = np.where(arr == x)
print(w)
arr[w] = x*2
print(arr)

# def himmelblau(X):
#     x, y = X
#     return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# def pega_inteiro(array):
#     my_str = "["
#     for item in array:
#         my_str += format(item, '.16f') + ", "
#     my_str = my_str[:-2]
#     my_str += "]"
#     return my_str

# arr = np.array([3.5777419964833865, -1.8447276214407460])
# print(arr)
# print(pega_inteiro(arr))

# print(himmelblau([-2.8051162331593371, 3.1313429854831063]))

# C = np.array([[1,0],[0,2]])

# w_i_mut = 0.5
# w_m_mut = 0.3
# w_s_mut = 0.2
# x_gb_mut = np.array([-0.23, 1.22])
# velocidade = np.array([-0.3, -1])
# posicao = np.array([-0.83, 1.32])
# x_best = np.array([0.23, 1.12])

# parte1 = w_i_mut * velocidade
# parte2 = w_m_mut * (x_best - posicao)

# pre1_parte3 = (x_gb_mut - posicao)
# pre2_parte3 = np.array([[pre1_parte3[0]],[pre1_parte3[1]]])

# pre3_parte3 = C @ pre2_parte3
# pre4_parte3 = np.array([pre3_parte3[0][0], pre3_parte3[1][0]])

# parte3 = w_s_mut * pre4_parte3
# velocidade = parte1 + parte2 + parte3
# print(velocidade)

# lista = np.random.normal(0, 1, size=1000)
# lista.sort()
# for item in lista:
#     print(item)