import bokeh
from bokeh import embed, io

from plotting import plot_conjugate


bokeh.io.output_file(filename='conjugate.html', title='conjugate')


fig = plot_conjugate()
script, div = embed.components(fig)

with open('abstract_math.html', 'r') as fin:
    htmlstring = fin.read()

newhtmlstring = htmlstring.replace("<!--PLOTSHOLDER-->", div).replace(
    "<!--SCRIPTHOLDER-->", script)

with open("index.html", "w") as fout:
    fout.write(newhtmlstring)
