def bisection_method(f, a, b, eps):
    """
    Метод деления пополам.
    f  – функция,
    [a,b] – исходный интервал, f(a)*f(b)<0,
    eps – точность по ширине отрезка.
    Возвращает список кортежей (i, a, b, x_mid, f(x_mid)).
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a) и f(b) должны иметь разные знаки")
    iterations = []
    i = 0
    while (b - a) >= eps:
        i += 1
        x_mid = (a + b) / 2.0
        fx = f(x_mid)
        iterations.append((i, a, b, x_mid, fx))
        if fx == 0:
            break
        if f(a) * fx < 0:
            b = x_mid
        else:
            a = x_mid
    return iterations
