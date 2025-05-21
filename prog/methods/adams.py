import math

def adams_moulton_system(f1, f2, x0, y10, y20, h, n):
    """
    Явно–неявный алгоритм Adams–Bashforth/Moulton 4-го порядка.
    Первые 3 шага — RK4 для разгона, далее предиктор–корректор.
    """
    def rk4_step(x, y1, y2):
        k1_1 = h * f1(x, y1, y2)
        k1_2 = h * f2(x, y1, y2)
        k2_1 = h * f1(x + h/2, y1 + k1_1/2, y2 + k1_2/2)
        k2_2 = h * f2(x + h/2, y1 + k1_1/2, y2 + k1_2/2)
        k3_1 = h * f1(x + h/2, y1 + k2_1/2, y2 + k2_2/2)
        k3_2 = h * f2(x + h/2, y1 + k2_1/2, y2 + k2_2/2)
        k4_1 = h * f1(x + h, y1 + k3_1, y2 + k3_2)
        k4_2 = h * f2(x + h, y1 + k3_1, y2 + k3_2)
        y1n = y1 + (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6
        y2n = y2 + (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6
        return x + h, y1n, y2n

    # разгон RK4
    xs, ys1, ys2 = [x0], [y10], [y20]
    for _ in range(3):
        x_prev, y1_prev, y2_prev = xs[-1], ys1[-1], ys2[-1]
        x_new, y1_new, y2_new = rk4_step(x_prev, y1_prev, y2_prev)
        xs.append(x_new); ys1.append(y1_new); ys2.append(y2_new)

    def get_f(i):
        return f1(xs[i], ys1[i], ys2[i]), f2(xs[i], ys1[i], ys2[i])

    # предиктор–корректор
    for i in range(3, n):
        f_im3, f_im2, f_im1, f_i = get_f(i-3), get_f(i-2), get_f(i-1), get_f(i)
        # Adams–Bashforth 4
        y1_pred = ys1[i] + h/24 * (55*f_i[0] - 59*f_im1[0] + 37*f_im2[0] - 9*f_im3[0])
        y2_pred = ys2[i] + h/24 * (55*f_i[1] - 59*f_im1[1] + 37*f_im2[1] - 9*f_im3[1])
        x_pred  = xs[i] + h

        # Adams–Moulton 4
        f_pred = f1(x_pred, y1_pred, y2_pred), f2(x_pred, y1_pred, y2_pred)
        y1_corr = ys1[i] + h/24 * (9*f_pred[0] + 19*f_i[0] - 5*f_im1[0] + f_im2[0])
        y2_corr = ys2[i] + h/24 * (9*f_pred[1] + 19*f_i[1] - 5*f_im1[1] + f_im2[1])

        xs.append(x_pred)
        ys1.append(y1_corr)
        ys2.append(y2_corr)

    return list(zip(xs, ys1, ys2))
