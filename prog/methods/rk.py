def runge_kutta_system(f1, f2, x0, y10, y20, h, n):
    results = []
    x, y1, y2 = x0, y10, y20

    for i in range(n + 1):
        results.append((x, y1, y2))

        k1_1 = h * f1(x, y1, y2)
        k1_2 = h * f2(x, y1, y2)

        k2_1 = h * f1(x + h/2, y1 + k1_1/2, y2 + k1_2/2)
        k2_2 = h * f2(x + h/2, y1 + k1_1/2, y2 + k1_2/2)

        k3_1 = h * f1(x + h/2, y1 + k2_1/2, y2 + k2_2/2)
        k3_2 = h * f2(x + h/2, y1 + k2_1/2, y2 + k2_2/2)

        k4_1 = h * f1(x + h, y1 + k3_1, y2 + k3_2)
        k4_2 = h * f2(x + h, y1 + k3_1, y2 + k3_2)

        y1 += (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6
        y2 += (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6
        x += h

    return results
