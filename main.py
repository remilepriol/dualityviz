import bokeh
import numpy as np
from bokeh import embed, io

from plotting import plot_conjugate


def entropy(x):
    return x * np.log(np.maximum(x, 1e-50)) + (1 - x) * np.log(np.maximum(1 - x, 1e-50))


def hinge(x):
    return np.maximum(1 - x, 0)


def squared_hinge(x):
    return np.maximum(1 - x, 0) ** 2


def random_polynomial(degree=4):
    coeffs = .3 * np.random.randn(degree + 1)
    coeffs[-1] = np.absolute(coeffs[-1])
    return np.polynomial.polynomial.Polynomial(coeffs)


bokeh.io.output_file(filename='conjugate.html', title='conjugate')


fig = plot_conjugate()
script, div = embed.components(fig)

with open('abstract_math.html', 'r') as fin:
    htmlstring = fin.read()

newhtmlstring = htmlstring.replace("<!--PLOTSHOLDER-->", div).replace(
    "<!--SCRIPTHOLDER-->", script)

with open("index.html", "w") as fout:
    fout.write(newhtmlstring)
