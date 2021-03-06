# Convex Duality Viz

A demo is available at <https://remilepriol.github.io/dualityviz> 

## How to run

Dependencies are python 3 and `numpy, matplotlib, bokeh`. 
To reproduce the plots contained in `index.html`, run
`python main.py`. 


## Roadmap

I want to create an interactive visualization of convex duality. 
Matplotlib does not support well interactive functionalities, and I want this visualization to be accessible to anyone in their browser.
The natural choice is Javascript d3, but it requires learning javascript and this whole new library.
The simpler alternative on which I focus here is Bokeh, a python library which lets me use custom javascript callbacks. 

### Principle

* Define `ColumnDataSource` with the data (similar to pandas dataframe). 
* Feed in this data as a source to some plots.
* Define a callback function that is called when some events happen `bokeh.events.MouseMove` or `bokeh.events.Tap` (for mobile compatibility.
* This callback is written in plain JS `CustomJS(code)`[doc](https://docs.bokeh.org/en/latest/docs/user_guide/interaction/callbacks.html#customjs-for-hover), and it updates the data source of plots that are defined in python. The intervention is minimal: updating tables of values. 
    * to debug javascript, open the chrome debugger in the output HTML file. 
* To make more complex interactions, I need to do a lot of the maths in javascript, but the plotting is still defined in python. Eventually, I would like to port everything to Javascript with D3.


### Content

* 6 plots 
    * $f(x)$ primal
    * $fc(g) = f^*(g)$ dual (f conjugate)
    * gradients $f'(x)$ and $fc'(g)$ (transposed)
    * $x.g-f(x)$ primal contour
    * $x.g - fc(g)$ dual contour
    * $f(x) + fc(g) - x.g$ Young's duality gap contour
* when I hover over each plot, it shows the corresponding point or line in the other plots.
    * [x] in the dual plot $(g,f^*(g))$ it shows the corresponding tangent of slope g in the primal plot, along with all the primal points $(x,f(x))$ that lie on this tangent.
    * [x] add vertical line from (0,0) to (0,linear approximation) to highlight the construction of the causal model.
    * [x] in the primal plot it shows the convex enveloppe reconstruction from the dual function.
    * [x] in the heatmap $(x,g)$,  it shows in the primal the line at coordinate $(x,f(x))$ with slope $g$. In the dual $(g, f^*(g))$ with slope $x$. 
* I add sliders to control [scaling properties](https://en.wikipedia.org/wiki/Convex_conjugate#Scaling_properties)
    * [x] x-shift of the primal adds a linear function to dual, $h(x) = f(x-x_0) \implies h^*(g) = f^*(g) + g.x_0$ 
    * [x] y-shift of the primal adds the opposite to the dual, $h(x) = f(x) + y_0 \implies h^*(g) = f^*(g) - y_0$ 
    * [x] adding a linear function to the primal causes an x-shift in the dual $h(x) = f(x)+g_0.x \implies h^*(g) = f^*(g - g_0)$
    * [x] x-axis dilation of the primal causes x-contration in the dual $h(x) = f(a.x) \implies h^*(g) = f^*(g/a)$
    * [x] y-axis dilation of the primal causes the same dilation in the dual, along with a contraction of the x-axis $h(x) = b.f(x) \implies h^*(g) = b.f^*(g/b)$

### Active  Ideas
* [ ] illustrate the saddle point surface of Lagrangian problems, as well as Fenchel problem.
* [ ] Why is the primitive of the reciprocal of the gradient equal to the fenchel conjugate, eg, the border of the dual epigraph ?
* [ ] generalization of convex duality with multifunction. Or in general dual of an arbitrary curve
* [ ] Can I illustrate infimal convolutions, and why Legendre is a group homomorphism to the addition. 

### Sleeping Ideas
* dual of a 2d shape (support function contour levels)
* generalization of convex duality with infimal convolutions, wavelet style. Then do the scattering transform of convexity.
