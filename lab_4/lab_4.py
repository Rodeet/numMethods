import numpy as np


def iteration_method(a, eps=0.0001):
    # Метод итераций
    x_0 = np.array([1]*a.shape[0])
    x_1 = a @ x_0
    l_k = x_1[0]/x_0[0]
    x_k = x_1
    while True:
        x_k1 = a @ x_k
        l_k1 = x_k1[0] / x_k[0]
        x_k1 /= max(x_k1)
        if abs(l_k1 - l_k) <= eps:
            break
        l_k = l_k1
        x_k = x_k1
    return x_k1, l_k1


def rotation_method():
    # Метод вращений
    pass


if __name__ == "__main__":
    A = np.array([[5., 1, 2],
                  [1, 4, 1],
                  [2, 1, 3]])
    print("Исходная матрица:")
    print(A)
    print("\nМетод итерации:")
    vec, zn = iteration_method(A)
    print("Собственный вектор:")
    print([round(i, 4) for i in vec])
    print("Собственное значение:")
    print(round(zn, 4))
