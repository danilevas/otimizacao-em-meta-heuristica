# Define a função de rosenbrock
def rosenbrock_n(v, dim, a=1, b=100):
    if dim != len(v):
        raise Exception("Número de dimensões não bate com tamanho do vetor")
    valor = 0.
    for d in range(dim-1):
        valor += (a - v[d])**2 + b * (v[d+1] - v[d]**2)**2
    return valor 

print(rosenbrock_n([1.1,2,2,2,1,1,1], 7))