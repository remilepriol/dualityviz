import bokeh
import matplotlib.cm
import matplotlib.colors
import numpy as np
from bokeh import events, layouts, models, palettes, plotting
from bokeh.models import ColumnDataSource, CustomJS


def colormap2hexpalette(cmlist):
    """Convert a list of matplotlib colormap names into a dictionary of hex list.

    Useful to get uniform sequential colormaps of length 256 with a single hue
    that are compatible with Bokeh, as this is not provided by default.
    """
    hexpalettes = {}
    for cmname in cmlist:
        rgbalist = matplotlib.cm.get_cmap(cmname)(np.linspace(0.0, 1.0, 256))
        hexlist = [matplotlib.colors.to_hex(c) for c in rgbalist]
        hexpalettes[cmname] = hexlist
    return hexpalettes


def set_range(fig, xx, yy, delta=.1):
    deltax = delta * (max(xx) - min(xx))
    deltay = delta * (max(yy) - min(yy))
    fig.x_range = models.Range1d(min(xx) - deltax, max(xx) + deltax)
    fig.y_range = models.Range1d(min(yy) - deltay, max(yy) + deltay)


def get_conjugate(ff, xrange, grange):
    # 2D array with all the combinations
    gx = grange[:, np.newaxis] * xrange[np.newaxis, :]
    gxf = gx - ff
    idxopt = np.argmax(gxf, axis=1, )  # optimal x values
    fc = gxf[np.arange(grange.shape[0]), idxopt]  # f conjugate
    return fc, idxopt, gx


def legendre(funcp, xrange):
    """Return the convex conjugate of an arbitrary function

    :param funcp: function handle to the function you want to plot
    :param xrange: x values for the function
    """

    # evaluate the function
    ff = funcp(xrange)

    # compute its gradient
    grad = np.diff(ff) / np.diff(xrange)
    is_f_convex = np.all(np.diff(grad) >= 0)
    grad = np.concatenate(([grad[0]], grad))

    # get range of gradients, defining the domain of the dual
    mg = min(grad)
    Mg = max(grad)
    grange = np.linspace(mg - 0.2 * (Mg - mg), Mg + 0.2 * (Mg - mg), len(xrange))

    # get f conjugate fc
    fc, idxopt, gx = get_conjugate(ff, xrange, grange)

    # get f conjugate conjugate fcc, aka convex enveloppe of f
    fcc, idgopt, _ = get_conjugate(fc, grange, xrange)
    fccgrad = np.diff(fcc) / np.diff(xrange)
    fccgrad = np.concatenate(([fccgrad[0]], fccgrad))

    return (xrange, ff, grad, fcc, fccgrad, idgopt,
            grange, fc, idxopt, gx, is_f_convex)


# import numeric.js
# https://cdnjs.cloudflare.com/ajax/libs/numeric/1.2.6/numeric.min.js

def plot_conjugate(funcp, xx):
    xx, ff, grad, fcc, fccgrad, idgopt, gg, fc, idxopt, gx, is_f_convex = legendre(funcp, xx)

    #########
    # SOURCES
    #########
    # a source is made of columns with equal lengths
    # all the primal and dual ingredients go to their respective source
    primal_source = ColumnDataSource(
        data=dict(xx=xx, ff=ff, grad=grad, fcc=fcc,
                  idgopt=idgopt, gopt=gg[idgopt]))
    dual_source = ColumnDataSource(
        data=dict(gg=gg, fc=fc, idxopt=idxopt, xopt=xx[idxopt]))
    # for images it's a bit more complex.
    # each column has length 1.
    # The image itself is an element
    # along with coordinates and width
    imagesdict = {
        'g.x - f(x)': [gx - ff],
        'g.x - f*(g)': [gx - fc[:, np.newaxis]],
        'f(x)+f*(g)-g.x': [ff + fc[:, np.newaxis] - gx]
    }
    source2d = ColumnDataSource(data={
        **imagesdict,
        'x': [xx[0]], 'dw': [xx[-1] - xx[0]],
        'y': [gg[0]], 'dh': [gg[-1] - gg[0]]
    })

    # COLORS
    palette = palettes.Category10_10  # palettes.Colorblind7
    primalcolor = palette[0]  # blue
    dualcolor = palette[1]  # orange
    envelopecolor = palette[2]
    tangentcolor = palette[3]
    heightcolor = palette[4]

    monochromemaps = colormap2hexpalette(['Oranges', 'Blues', 'Purples'])

    ########
    # GLYPHS
    ########
    # global options for the figures
    pixelsize = 250
    opts = dict(plot_width=pixelsize, plot_height=pixelsize)
    # tools='pan,wheel_zoom',active_scroll='wheel_zoom')

    # plot the primal function
    fig1 = plotting.figure(title='Primal f(x)', **opts,
                           x_axis_label='x', y_axis_label='y')
    set_range(fig1, xx, ff)

    if not is_f_convex:
        envelope = fig1.line('xx', 'fcc', source=primal_source, line_width=2,
                             color=envelopecolor)
    primal = fig1.line('xx', 'ff', source=primal_source, line_width=3,
                       color=primalcolor)

    primaltangent = fig1.line('x', 'y', line_width=1, color=tangentcolor, alpha=.7,
                              source=ColumnDataSource(dict(x=[], y=[])))
    primalpoint = fig1.circle('x', 'y', size=10, color=tangentcolor,
                              source=ColumnDataSource(dict(x=[], y=[])))
    primalheight = fig1.line('x', 'y', line_width=3, color=heightcolor,
                             source=ColumnDataSource(dict(x=[], y=[])))

    # plot the conjugate function
    fig2 = plotting.figure(title='Dual f*(g)', **opts,
                           x_axis_label='g', y_axis_label='y')
    set_range(fig2, gg, fc)
    fig2.line('gg', 'fc', source=dual_source, line_width=3, color=dualcolor)

    dualpoint = fig2.circle('g', 'y', size=10, color=tangentcolor, alpha=.7,
                            source=ColumnDataSource(dict(g=[], y=[])))
    dualtangent = fig2.line('g', 'y', line_width=1, color=tangentcolor,
                            source=ColumnDataSource(dict(g=[], y=[])))
    dualheight = fig2.line('g', 'y', line_width=3, color=heightcolor,
                           source=ColumnDataSource(dict(g=[], y=[])))

    # highlight lines x=0 and y=0 in primal and dual plots
    for fig in [fig1, fig2]:
        for dimension in [0, 1]:
            grid0 = models.Grid(dimension=dimension, grid_line_color='black',
                                ticker=models.FixedTicker(ticks=[0]))
            fig.add_layout(grid0)

    # displaying information on primal plot
    # infolabel = models.Label(text='', x=30, y=100, x_units='screen', y_units='screen')
    # fig1.add_layout(infolabel)

    # plot gradients
    fig3 = plotting.figure(title='Derivatives', **opts,
                           x_axis_label='x', y_axis_label='g')
    fig3.line('xopt', 'gg', source=dual_source, line_width=3, color=dualcolor)
    fig3.line('xx', 'grad', source=primal_source, color=primalcolor, line_width=3)

    # HEAT MAPS
    images = [plotting.figure(**opts, x_axis_label='x', y_axis_label='g') for _ in range(3)]
    for fig, name, colormap in zip(images, imagesdict, monochromemaps.values()):
        fig.title.text = name
        fig.image(image=name, x='x', dw='dw', y='y', dh='dh', alpha=.7,
                  source=source2d, palette=colormap)
    lw = 2
    images[0].line('xx', 0, source=primal_source, color=primalcolor, line_width=lw)
    images[0].line('xopt', 'gg', source=dual_source, color=dualcolor, line_width=lw)

    images[1].line(0, 'gg', source=dual_source, color=dualcolor, line_width=lw)
    images[1].line('xx', 'gopt', source=primal_source,
                   color=primalcolor if is_f_convex else envelopecolor, line_width=lw)

    gxpoint_source = ColumnDataSource(data=dict(x=[], g=[]))
    gxline_source = ColumnDataSource(data=dict(x=[], g=[]))
    for fig in [fig3] + images:
        fig.circle(x='x', y='g', size=10, color=tangentcolor, alpha=.6, source=gxpoint_source)
        fig.line(x='x', y='g', line_width=lw, line_color=tangentcolor, source=gxline_source)

    ##############
    # INTERACTIONS
    ##############
    # SLIDERS
    # Scaling the primal with 5 sliders
    deltax = (max(xx) - min(xx))
    deltay = (max(ff) - min(ff))
    deltag = (max(gg) - min(gg))
    sliders_dict = {
        'x_shift': models.Slider(
            start=-deltax, end=deltax, value=0, step=deltax / 100, title="x shift"
        ),
        'f_shift': models.Slider(
            start=-deltay, end=deltay, value=0, step=deltay / 100, title="f shift"
        ),
        'g_shift': models.Slider(
            start=-deltag, end=deltag, value=0, step=deltag / 100, title="g shift"
        ),
        'x_dilate': models.Slider(
            start=.5, end=2, value=1, step=0.01, title="x dilation"
        ),
        'f_dilate': models.Slider(
            start=.5, end=2, value=1, step=0.01, title="f dilation"
        )
    }

    slider_callback = CustomJS(
        args={
            'xx': xx, 'ff': ff, 'gg': gg, 'fc': fc,
            'primal': primal_source, 'dual': dual_source,
            'source2d': source2d, **sliders_dict
        },
        code="""
        let dx = x_shift.value;
        f_shift.value
        g_shift.value
        x_dilate.value
        f_dilate.value
        
        for (let i=0; i<primal.data['xx'].length ;i++){
            primal.data['xx'][i] += dx ;
        }
        primal.change.emit();

        for (let i=0; i<dual.data['gg'].length ;i++){
            dual.data['fc'][i] += dual.data['gg'][i]*dx;
            dual.data['xopt'][i] += dx;
        }         
        dual.change.emit();

        source2d.data['x'][0] += dx;
        source2d.change.emit();
        """
    )
    x_dilate_slider.js_on_change('value', CustomJS(
        args=slider_args,
        code="""
        console.log(cb_obj,cb_obj.new,cb_obj.old,cb_obj.attr)
        let previous_x_dilate = primal.data['xx'][1]/xx[1];
        let dx = cb_obj.value / previous_x_dilate;
        for (let i=0; i<primal.data['xx'].length ;i++){
            primal.data['xx'][i] *= dx ;
        }
        primal.change.emit();

        for (let i=0; i<dual.data['gg'].length ;i++){
            dual.data['gg'][i] /= dx;
            dual.data['xopt'][i] += dx;
        }         
        dual.change.emit();

        source2d.data['x'][0] += dx;
        source2d.change.emit();
        """
    ))
    y_shift_slider.js_on_change('value', CustomJS(
        args=slider_args,
        code="""
        let previous_y_shift = primal.data['ff'][0]-ff[0];
        let dy = cb_obj.value - previous_y_shift;
        for (let i=0; i<primal.data['xx'].length ;i++){
            primal.data['ff'][i] += dy;
            primal.data['fcc'][i] += dy;
        }
        primal.change.emit();

        for (let i=0; i<dual.data['gg'].length ;i++){
            dual.data['fc'][i] -= dy;
        }         
        dual.change.emit();
        """
    ))
    g_shift_slider.js_on_change('value', CustomJS(
        args=slider_args,
        code="""
        let previous_g_shift = dual.data['gg'][0]-gg[0];
        let dg = cb_obj.value - previous_g_shift;
        for (let i=0; i<primal.data['xx'].length ;i++){
            primal.data['ff'][i] += dg * primal.data['xx'][i];
            primal.data['fcc'][i] += dg * primal.data['xx'][i];
            primal.data['grad'][i] += dg;
            primal.data['gopt'][i] += dg;
        }
        primal.change.emit();

        for (let i=0; i<dual.data['gg'].length ;i++){
            dual.data['gg'][i] += dg;
        }         
        dual.change.emit();
        
        source2d.data['y'][0] += dg;
        source2d.change.emit();
        """
    ))

    for slider in sliders_dict:
        slider.js_on_change('value', slider_callback)

    # HOVERING
    hover_source_dict = {
        'tangent': primaltangent.data_source,
        'point': primalpoint.data_source,
        'primalheight': primalheight.data_source,
        'dualpoint': dualpoint.data_source,
        'dualtangent': dualtangent.data_source,
        'dualheight': dualheight.data_source,
        'gxpoint': gxpoint_source,
        'gxline': gxline_source
    }

    source_dict = {
        'primal': primal_source,
        'dual': dual_source,
        **hover_source_dict
    }

    # hover over primal plot
    fig1.js_on_event(bokeh.events.MouseMove, CustomJS(
        args=source_dict,
        code="""
        let x0 = cb_obj.x;
        let xx = primal.data['xx'];
        let i = xx.findIndex(x => x >=x0);
        if (i==-1){i = xx.length-1};
        let x1 = xx[i];
        let y1 = primal.data['fcc'][i];
        point.data['x'] = [x1];
        point.data['y'] = [y1];
        point.change.emit();
        primalheight.data['x'] = [x1, x1];
        primalheight.data['y'] = [0, y1];
        primalheight.change.emit();

        let j = primal.data['idgopt'][i];
        let gg = dual.data['gg'];
        let g1 = gg[j];
        let fc1 = dual.data['fc'][j];        
        dualpoint.data['g'] = [g1];
        dualpoint.data['y'] = [fc1];
        dualpoint.change.emit();

        dualheight.data['g'] = [0,0];
        dualheight.data['y'] = [0,fc1 - g1*x1];
        dualheight.change.emit();

        dualtangent.data['g'] = [gg[0]-1000, gg[gg.length-1]+1000];
        dualtangent.data['y'] = dualtangent.data['g'].map(g => fc1 + x1*(g-g1));
        dualtangent.change.emit();

        gxpoint.data['x'] = [x1];
        gxpoint.data['g'] = [g1];
        gxpoint.change.emit();   
             
        gxline.data['x'] = [x1, x1];
        gxline.data['g'] = [gg[0], gg[gg.length - 1]];
        gxline.change.emit();
        """
    ))

    # hover over dual plot
    fig2.js_on_event(bokeh.events.MouseMove, CustomJS(
        args=source_dict,
        code="""
        let g0 = cb_obj.x;
        let gg = dual.data['gg'];
        let j = gg.findIndex(g => g >=g0);
        if (j==-1){j = gg.length-1};
        let g1 = gg[j];
        let fc1 = dual.data['fc'][j];
        dualpoint.data['g'] = [g1];
        dualpoint.data['y'] = [fc1];
        dualpoint.change.emit();

        dualheight.data['g'] = [g1, g1];
        dualheight.data['y'] = [0, fc1];
        dualheight.change.emit();

        let i = dual.data['idxopt'][j];
        let xx = primal.data['xx'];
        let x1 = xx[i];
        let y1 = primal.data['ff'][i];        
        point.data['x'] = [x1];
        point.data['y'] = [y1];
        point.change.emit();

        tangent.data['x'] = [xx[0]-1000, xx[xx.length-1]+1000];
        tangent.data['y'] = tangent.data['x'].map(x => g1*(x-x1) + y1);
        tangent.change.emit();

        primalheight.data['x'] = [0,0];
        primalheight.data['y'] = [0,y1 - g1*x1];
        primalheight.change.emit();

        gxpoint.data['x'] = [x1];
        gxpoint.data['g'] = [g1];
        gxpoint.change.emit();        
        
        gxline.data['x'] = [xx[0], xx[xx.length - 1]];
        gxline.data['g'] = [g1, g1];
        gxline.change.emit();
        """
    ))

    # Remove all temporary glyphs on MouseLeave
    jsleave = CustomJS(
        args={'sourcelist': list(hover_source_dict.values())},
        code="""
        for(let source of sourcelist){
            for(let key in source.data){
                source.data[key]=[];
            }
            source.change.emit();
        }
        """
    )
    for fig in [fig1, fig2, fig3] + images:
        fig.js_on_event(bokeh.events.MouseLeave, jsleave)

    # bigfig = layouts.gridplot([[fig1,fig2,fig3]], toolbar_location='')
    bigfig = layouts.gridplot([[x_dilate_slider, None, None],
                               [x_shift_slider, y_shift_slider, g_shift_slider],
                               [fig1, fig2, fig3],
                               images], toolbar_location=None)
    return bigfig
