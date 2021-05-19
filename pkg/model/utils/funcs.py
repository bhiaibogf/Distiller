def sqr(a):
    return a * a


def quick_pow(a, n):
    if n == 1:
        return a
    elif n == 2:
        return sqr(a)

    a_n_2 = quick_pow(a, n // 2)
    if n % 2 == 0:
        return sqr(a_n_2)
    else:
        return sqr(a_n_2) * a


def mon2lin(color):
    return color.pow(2.2)


def mix(x, y, a):
    return x * (1 - a) + y * a
