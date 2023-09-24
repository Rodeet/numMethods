def f(cfs, x):
    val = 0
    for i, item in enumerate(cfs[::-1]):
        val += item * x ** i
    return val


def derivative(cfs, x):
    k = []
    for i, item in enumerate(cfs[::-1]):
        k.append(item * i)
    k = k[1:]
    val = 0
    for i, item in enumerate(k):
        val += item * x ** i
    return val


def newton_simplified(cfs, x0, eps=0.01, x_n=None):
    if not x_n:
        x_n = x0
    x_n1 = x_n - f(cfs, x_n) / derivative(cfs, x0)
    if abs(x_n1 - x_n) < eps:
        return x_n1
    else:
        return newton_simplified(cfs, x0, eps, x_n1)


def newton_broyden(cfs, x0, eps=0.01, c=1.00):
    x_n = x0 - c * f(cfs, x0) / derivative(cfs, x0)
    if abs(x_n - x0) < eps:
        return x_n
    else:
        return newton_broyden(cfs, x_n, eps, c)


def dif(cfs, x1, x0):
    return (f(cfs, x1) - f(cfs, x0)) / (x1 - x0)


def secant(cfs, x0, eps=0.01, sig=0.01, x2=None):
    if not x2:
        x2 = x0 - sig
    x_n1 = x0 - f(cfs, x0) / dif(cfs, x0, x2)
    if abs(x_n1 - x0) < eps:
        return x_n1
    else:
        return secant(cfs, x_n1, x2=x0)


def theorem_3(cfs):
    a = max(list(map(abs, cfs[1:])))
    b = max(list(map(abs, cfs[:-1])))
    left = 1 / (1 + b / abs(cfs[-1]))
    right = 1 + a / abs(cfs[0])
    return left, right


def theorem_4(cfs):
    if cfs[0] < 0:
        cfs = [i*-1 for i in cfs]
    c = 0
    index = -1
    for i, item in enumerate(cfs):
        if item < 0:
            if index == -1:
                index = len(cfs) - i - 1
            if abs(item) > c:
                c = abs(item)
    r = 1 + (c / cfs[0]) ** (1 / (len(cfs) - 1 - index))
    return r


def theorem_5(cfs):
    r = theorem_4(cfs)
    r1 = theorem_4(cfs[::-1])
    k = [item*-1 if (len(cfs) - i) % 2 == 0 else item for i, item in enumerate(cfs)]
    r2 = theorem_4(k)
    r3 = theorem_4(k[::-1])
    left_p = 1/r1
    right_p = r
    left_o = -r2
    right_o = -1/r3
    return left_p, right_p, left_o, right_o


if __name__ == "__main__":
    print("Введите коэффициенты уравнения через пробел (an an-1 ... a0:")
    coefs = list(map(float, input().split()))
    print("Введите x0:")
    x0 = float(input())
    print("Теорема 3. Оценка модулей корней уравнения")
    left_b, right_b = theorem_3(coefs)
    print("%.2f < x_i+ <= %.2f" % (left_b, right_b))
    print("%.2f <= x_i- < %.2f" % (-right_b, -left_b))
    print("Теорема 4. Теорема Лагранжа о верхней границе положительных корней")
    top_b = theorem_4(coefs)
    print("x_i+ <= %.2f" % right_b)
    print("Теорема 5. Теорема о верхней и нижней границе положительных и отрицательных корней")
    l1, r1, l2, r2 = theorem_5(coefs)
    print("%.2f < x_i+ <= %.2f" % (l1, r1))
    print("%.2f <= x_i- < %.2f" % (l2, r2))
    print("1. Метод Ньютона")
    print("x = %.3f" % newton_broyden(coefs, x0))
    print("2. Метод Ньютона (упрощённый)")
    print("x = %.3f" % newton_simplified(coefs, x0))
    print("3. Метод Ньютона-Бройдена (c=0.5)")
    print("x = %.3f" % newton_broyden(coefs, x0, c=0.5))
    print("4. Метод Ньютона-Бройдена (c=1.5)")
    print("x = %.3f" % newton_broyden(coefs, x0, c=1.5))
    print("5. Метод секущих")
    print("x = %.3f" % secant(coefs, x0))
