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

np.random.seed(0)
functiondict = {
    '|x|': (np.abs, -1, 1),
    'x^4 ': (lambda x: x ** 4, -1, 1),
    'x^4 + 4 x^3 ': (lambda x: x ** 4 + 4 * x ** 3, -1, 1),
    'x^2 - \cos(2 x)': (lambda x: x ** 2 - np.cos(2 * x), -5, 5),
    #
    # 'sin(x)': (np.sin, -np.pi, 0),
    # 'Entropy(x) = x log(x) + (1-x) log(1-x)': (entropy, 0, 1),
    # 'x sin(1/x)': (lambda x: x * np.sin(1 / x), -1, 1),
    # '0': (np.zeros_like, -1, 1), # not working yet
    # 'Hinge(x) = max(0,1-x)': (hinge, -2, 2),
    # 'Hinge(x)**2': (squared_hinge, -2, 2),
    # 'Random Polynomial of degree 6': (random_polynomial(6), -2, 2),
}

# abstract=''
# with open('abstract.txt', 'r') as file:
#     abstract = file.read()
# figlist = [models.Div(text='<h1>Convex Conjugate Visualization</h1> \n' + abstract)]

figlist = []
scriptlist = []
for name, (func, inf, sup) in functiondict.items():
    figtitle = f"<h2 class='plot-title'> \( x \mapsto {name} ; x \in [{inf}, {sup}] \)</h2>"
    fig = plot_conjugate(func, np.linspace(inf, sup, 150))
    script, div = embed.components(fig)
    figlist += [figtitle, div]
    scriptlist += [script]

figstring = '\n'.join(figlist)
script = '\n'.join(scriptlist)

with open('abstract_math.html', 'r') as fin:
    htmlstring = fin.read()

newhtmlstring = htmlstring.replace("PLOTSHOLDER", figstring).replace(
    "<div>SCRIPTHOLDER</div>", script)

with open("index.html", "w") as fout:
    fout.write(newhtmlstring)

# bokeh.io.save(finalplot, title='conjugates', filename='conjugate.html', )
