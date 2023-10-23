import numpy as np


def reversal(m):
    res = np.array([1] * (len(m) + 1))
    for i in range(-1, -len(m) - 1, -1):
        a = m[i] * res
        res[i - 1] = (a[-1] - a[-2:i - 1:-1].sum()) / a[i - 1]
    return res[:-1]


def gauss_div(m):
    for i in range(len(m)-1):
        m[i] = m[i] / m[i, i]
        for j in range(i+1, len(m)):
            m[j] -= m[i]*m[j, i]
    return reversal(m)


def gauss_rectangle(m):
    r, c = m.shape
    for k in range(len(m)):
        for i in range(k+1, r):
            for j in range(k+1, c):
                m[i, j] -= m[k, j]*m[i, k]/m[k, k]
        m[k + 1:, k] *= 0
    return reversal(m)


def gauss_lead(m):
    for i in range(len(m)-1):
        row = np.argmax(np.abs(m[i:, i]))
        m[[i, row+i]] = m[[row+i, i]]
        m[i] = m[i] / m[i, i]
        for j in range(i+1, len(m)):
            m[j] -= m[i]*m[j, i]
    return reversal(m)


def simple_iter(a, b, eps=0.001):
    x = np.array([1.0] * (len(b)))

    while True:
        x_new = np.copy(x)
        for i in range(len(b)):
            s = 0
            for j in range(len(b)):
                if j != i:
                    s += a[i][j]*x[j]
            x[i] = (b[i] - s)/a[i][i]

        if np.linalg.norm(x_new - x) <= eps:
            break
    return x


def zeidel(a, b, eps=0.001):
    n = len(a)
    x = np.zeros(n)

    while True:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(a[i][j] * x_new[j] for j in range(i))
            s2 = sum(a[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / a[i][i]

        x = x_new
        if np.linalg.norm(x_new - x) <= eps:
            break
    return x


if __name__ == "__main__":
    print("1. Метод Гаусса (единичного деления)")
    matrix_1 = np.array([[2.0, 1, 4, 16],
                         [3, 2, 1, 10],
                         [1, 3, 3, 16]])
    print("Исходная матрица:")
    print(matrix_1)
    print("Ответ:")
    ans = gauss_div(matrix_1)
    for n, item in enumerate(ans):
        if n+1 == len(ans):
            print(f"x{n + 1} = {item}")
        else:
            print(f"x{n + 1} = {item}", end=", ")
    print("2. Метод Гаусса (правило прямоугольника)")
    matrix_2 = np.array([[2.0, 1, 4, 16],
                         [3, 2, 1, 10],
                         [1, 3, 3, 16]])
    print("Исходная матрица:")
    print(matrix_2)
    print("Ответ:")
    ans = gauss_rectangle(matrix_2)
    for n, item in enumerate(ans):
        if n + 1 == len(ans):
            print(f"x{n + 1} = {item}")
        else:
            print(f"x{n + 1} = {item}", end=", ")
    print("3. Метод Гаусса (Выбор ведущего элемента по столбцам)")
    matrix_3 = np.array([[-3, 2.099, 6, 3.901],
                         [10, -7, 0, 7],
                         [5, -1, 5, 6]])
    print("Исходная матрица:")
    print(matrix_3)
    print("Ответ:")
    ans = gauss_lead(matrix_3)
    for n, item in enumerate(ans):
        if n + 1 == len(ans):
            print(f"x{n + 1} = {item}")
        else:
            print(f"x{n + 1} = {item}", end=", ")
    print("4. Метод простых итераций")
    a_1 = np.array([[10.0, 1, -1],
                    [1, 10, -1],
                    [-1, 1, 10]])
    b_1 = np.array([11.0, 10, 10])
    print("Ответ:")
    ans = simple_iter(a_1, b_1)
    for n, item in enumerate(ans):
        if n + 1 == len(ans):
            print(f"x{n + 1} = {round(item, 2)}")
        else:
            print(f"x{n + 1} = {round(item, 2)}", end=", ")
    print("5. Метод Зейделя")
    a_2 = np.array([[20.9, 1.2, 2.1, 0.9],
                    [1.2, 21.2, 1.5, 2.5],
                    [2.1, 1.5, 19.8, 1.3],
                    [0.9, 2.5, 1.3, 32.1]])
    b_2 = np.array([[21.7],
                    [27.46],
                    [28.76],
                    [49.72]])
    print("Ответ:")
    ans = zeidel(a_2, b_2)
    for n, item in enumerate(ans):
        if n + 1 == len(ans):
            print(f"x{n + 1} = {round(item, 2)}")
        else:
            print(f"x{n + 1} = {round(item, 2)}", end=", ")
