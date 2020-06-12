# Visualizing the Convex Conjugate

Given a function f from real numbers to real numbers, its Legendre-Fenchel transformation is defined by the formula f*(g) = max_x g.x - f(x).  This transformation is a ubiquitous concept throughout Science. It appears in thermodynamics and classical mechanics as the Legendre transform (a special case), in convex optimization and machine learning as the Fenchel conjugate or the convex conjugate. At first sight this definition seems arbitrary, but it admits geometrical interpretations along with many properties that make it a useful tool. Among others, it offers a generalization of Lagrangian duality.

Unfortunately understanding these properties typically require newcomers to assimilate numerous formulas. So far I have found no widely accessible visualization to help one gain an intuitive knowledge of this transformation. I tried to fill this gap with an interactive visualization that runs in the browser. I programmed it with a combination of the open source Python library Bokeh and custom JavaScript callbacks.

This code takes a real function f defined on a certain range [x0,x1], and defined to be plus infinity outside this range. (I know it sounds exotic but it's a convention in convex analysis.) The code displays 6 functions defined over three dimensions: the X, G and Y spaces. Note that G stands for 'gradient' because of the higher dimensional generalization.

## 6 plots

- The upper left axis shows the function f: X -> Y -- along with its convex envelope if f is not convex. Note that the convex envelope is equal to the  bi-conjugate f**.
- The upper central axis, shows the conjugate f\*; G -> Y.
- The upper right axis shows the (sub-)differential of f, df: X -> G, along with df\*: G -> X. If f is strictly convex and differentiable then these two are monotonous, bijective and reciprocal mappings: f\*'(f'(x))=x. This is the special case of the Legendre transform : to get f\*, differentiate f, take the reciprocal and integrate. In general g(x)=df\*\*(x) and x(g)= df\*(g) where equalities are between sets.
- The lower row shows mappings from the (X, G) plane to Y.
    - The left axis shows the landscape used to build f\*. Slicing this landscape along the blue line gives -f and the orange line gives f*.
    - The central axis shows the landscape to build f. Slicing along the orange line gives -f\* and the blue (or green) line gives f**.
    - The right axis shows Young's duality gap f(x) + f\*(g) - x.g which is  non-negative by definition of f\* and zero iff x=x(g) or g=g(x).

## Interactivity

**Hovering** over the dual plot shows how f* is built via f\*(g) = max_x g.x - f(x) = max_x g.x - f\*\*(x), highlighting a primal point (x(g), f(x(g))) which achieves this max. A similar behavior is encoded in the primal plot, building f\*\* from f\*.

**Sliders** illustrate the scaling properties of f and f\*.

We hope the reader will be able to figure out the meaning of all plot elements by playing with it.
