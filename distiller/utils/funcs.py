import torch


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


def float2hex(x):
    return str(hex(int(round(x * 255))))


def to_hex(color):
    color = list(map(float2hex, color))
    html_color = ''
    for cl in color:
        hex_color = cl[2:]
        html_color += '0' * (2 - len(hex_color)) + hex_color
    return html_color


def batch_vec_dot(a, b):
    return torch.tensordot(a, b, dims=([1], [1])).squeeze()


def batch_scale(num, vec):
    """
    :parameter: num ([batch])
    :parameter: vec ([dim])
    """
    return (num.unsqueeze(1) @ vec.unsqueeze(0)).squeeze()
