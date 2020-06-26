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
    hexpalettes = []
    for cmname in cmlist:
        rgbalist = matplotlib.cm.get_cmap(cmname)(np.linspace(0.0, 1.0, 256))
        hexlist = [matplotlib.colors.to_hex(c) for c in rgbalist]
        hexpalettes += [hexlist]
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
    mg = min(grad) - .1
    Mg = max(grad) + .1
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

def plot_conjugate(funcp, xx, pixelsize=350):
    xx, ff, grad, fcc, fccgrad, idgopt, gg, fc, idxopt, gx, is_f_convex = legendre(funcp, xx)

    #########
    # SOURCES
    #########
    # a source is made of columns with equal lengths
    # all the primal and dual ingredients go to their respective source
    primal_source = ColumnDataSource(data=dict(
        xx=xx, ff=ff, grad=grad, fcc=fcc,
        idgopt=idgopt, gopt=gg[idgopt], gzeros=np.zeros_like(xx)
    ))
    # note to self : gopt is also the gradient of f**
    dual_source = ColumnDataSource(data=dict(
        gg=gg, fc=fc, idxopt=idxopt,
        xopt=xx[idxopt], xzeros=np.zeros_like(gg)
    ))
    # for images it's a bit more complex.
    # each column has length 1.
    # The image itself is an element
    # along with coordinates and width
    images_dict = {
        'gxminusf': [gx - ff],
        'gxminusfc': [gx - fc[:, np.newaxis]],
        'youngs': [ff + fc[:, np.newaxis] - gx]
    }
    source2d = ColumnDataSource(data={
        **images_dict,
        'x0': [xx[0]], 'delta_x': [xx[-1] - xx[0]],
        'g0': [gg[0]], 'delta_g': [gg[-1] - gg[0]]
    })
    image_titles = {'gxminusf': 'g.x - f(x) (maximize on x to get f*)',
                    'gxminusfc': 'g.x - f*(g) (maximize on g to get f**)',
                    'youngs': "Young's duality gap f(x)+f*(g)-g.x"}

    # COLORS
    palette = palettes.Category10_10  # palettes.Colorblind7
    primalcolor = palette[0]  # blue
    dualcolor = palette[1]  # orange
    gapcolor = palette[2]  # green
    tangentcolor = palette[3]  # red
    heightcolor = palette[4]  # purple

    monochromemaps = colormap2hexpalette(['Purples', 'Purples', 'Greens'])

    ########
    # GLYPHS
    ########
    # global options for the figures
    opts = dict(plot_width=pixelsize, plot_height=pixelsize)

    # plot the primal function
    fig1 = plotting.figure(title='Primal f(x)', **opts, tools='pan',
                           # tools='pan,wheel_zoom', active_scroll='wheel_zoom',
                           x_axis_label='x', y_axis_label='y')
    set_range(fig1, xx, ff)

    if not is_f_convex:
        fig1.line('xx', 'fcc', source=primal_source, line_width=3, color=primalcolor, alpha=.5)
    fig1.line('xx', 'ff', source=primal_source, line_width=3, color=primalcolor)

    primalpoint = fig1.circle('x', 'y', size=10, color=tangentcolor,
                              source=ColumnDataSource(dict(x=[], y=[])))
    primaltangent = fig1.line('x', 'y', line_width=2, color=tangentcolor,
                              source=ColumnDataSource(dict(x=[], y=[])))
    primaldroite = fig1.line('x', 'y', line_width=2, color='black',
                             source=ColumnDataSource(dict(x=[], y=[])))
    primalheight = fig1.line('x', 'y', line_width=3, color=heightcolor, line_cap='round',
                             source=ColumnDataSource(dict(x=[], y=[])))
    primalgap = fig1.line('x', 'y', line_width=3, color=gapcolor, line_dash='dotted',
                          source=ColumnDataSource(dict(x=[], y=[])))

    # plot the conjugate function
    fig2 = plotting.figure(title='Dual f*(g)', **opts,
                           tools='pan', x_axis_label='g', y_axis_label='y')
    set_range(fig2, gg, fc)
    fig2.line('gg', 'fc', source=dual_source, line_width=3, color=dualcolor)

    dualpoint = fig2.circle('g', 'y', size=10, color=tangentcolor, alpha=.7,
                            source=ColumnDataSource(dict(g=[], y=[])))
    dualtangent = fig2.line('g', 'y', line_width=2, color=tangentcolor,
                            source=ColumnDataSource(dict(g=[], y=[])))
    dualdroite = fig2.line('g', 'y', line_width=2, color='black',
                           source=ColumnDataSource(dict(g=[], y=[])))
    dualheight = fig2.line('g', 'y', line_width=3, color=heightcolor, line_cap='round',
                           source=ColumnDataSource(dict(g=[], y=[])))
    dualgap = fig2.line('g', 'y', line_width=3, color=gapcolor, line_dash='dotted',
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
    fig3 = plotting.figure(title='Derivatives', **opts, tools='',
                           x_axis_label='x', y_axis_label='g')
    fig3.line('xopt', 'gg', source=dual_source, line_width=3, color=dualcolor)
    fig3.line('xx', 'grad', source=primal_source, color=primalcolor, line_width=3)

    # IMAGES
    images = [plotting.figure(**opts, tools='', x_axis_label='x', y_axis_label='g') for _ in
              range(3)]
    for fig, name, colormap in zip(images, image_titles, monochromemaps):
        fig.title.text = image_titles[name]
        fig.image(image=name, x='x0', dw='delta_x', y='g0', dh='delta_g', alpha=.7,
                  source=source2d, palette=colormap)

    lw = 2
    images[0].line('xx', 'gzeros', source=primal_source, color=primalcolor, line_width=lw,
                   legend_label='-f(x)')
    images[0].line('xopt', 'gg', source=dual_source, color=dualcolor, line_width=lw,
                   legend_label='f*(g)')

    images[1].line('xzeros', 'gg', source=dual_source, color=dualcolor, line_width=lw,
                   legend_label='-f*(g)')
    images[1].line('xx', 'gopt', source=primal_source,
                   color=primalcolor, line_width=lw,
                   legend_label='f(x)')

    images[2].line('xx', 'gzeros', source=primal_source, color=primalcolor, line_width=lw,
                   legend_label='f(x)')
    images[2].line('xzeros', 'gg', source=dual_source, color=dualcolor, line_width=lw,
                   legend_label='f*(g)')
    images[2].line('xopt', 'gg', source=dual_source, color='black', alpha=.5, line_width=lw,
                   legend_label='0')

    for fig in images:
        fig.legend.background_fill_alpha = .1
        fig.legend.location = 'top_left'

    # temporary hover glyphs
    gxcircle = models.Circle(x='x', y='g', size=10, fill_color=tangentcolor,
                             line_color=tangentcolor, fill_alpha=.7)
    gxline = models.Line(x='x', y='g', line_width=lw, line_color=tangentcolor)

    hline = images[0].add_glyph(ColumnDataSource(data=dict(x=[], g=[])), gxline)
    hpoint = images[0].add_glyph(ColumnDataSource(data=dict(x=[], g=[])), gxcircle)
    vline = images[1].add_glyph(ColumnDataSource(data=dict(x=[], g=[])), gxline)
    vpoint = images[1].add_glyph(ColumnDataSource(data=dict(x=[], g=[])), gxcircle)

    for fig in [fig3, images[2]]:
        fig.add_glyph(hpoint.data_source, gxcircle)
        fig.add_glyph(vpoint.data_source, gxcircle)

    ##############
    # INTERACTIONS
    ##############
    # INPUT FUNCTION
    # very complicated without going full blown JS
    inputfunc = bokeh.models.TextInput(title='f(x)=', value='x^4')
    inputfunc.js_on_change('value', CustomJS(
        args={},
        code="""
            math.eval(cb_obj.value,x)
        """))

    # SLIDERS:  Scaling the primal with 5 sliders
    deltax = (max(xx) - min(xx)) + .1
    deltay = (max(ff) - min(ff)) + .1
    deltag = (max(gg) - min(gg)) + .1
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
            **sliders_dict,
            **primal_source.data, **dual_source.data, **source2d.data,
            'primal': primal_source, 'dual': dual_source, 'source2d': source2d
        },
        code="""
            const xs = x_shift.value;
            const fs = f_shift.value;
            const gs = g_shift.value;
            const xd = x_dilate.value;
            const fd = f_dilate.value;
            
            function xtransform(x){
                return x/xd + xs
            }
            function gtransform(g){
                return xd * fd * g + gs
            }
            
            let new_xx = xx.map(xtransform);
            let new_gg = gg.map(gtransform);
            
            for (let i=0; i<primal.data['xx'].length ;i++){
                primal.data['xx'][i] = new_xx[i];
                primal.data['gzeros'][i] = gtransform(0);
                primal.data['grad'][i] = gtransform(grad[i]);
                primal.data['gopt'][i] = gtransform(gopt[i]);
                primal.data['ff'][i] = fd * ff[i] + gs * (new_xx[i]-xs)  + fs;
                primal.data['fcc'][i] = fd * fcc[i] + gs * (new_xx[i]-xs)  + fs;
            }
            primal.change.emit();
    
            for (let i=0; i<dual.data['gg'].length ;i++){
                dual.data['xzeros'][i] = xtransform(0);
                dual.data['xopt'][i] = xtransform(xopt[i]) ;
                dual.data['gg'][i] = new_gg[i];
                dual.data['fc'][i] = fd*fc[i] + xs * new_gg[i] - fs;
            }         
            dual.change.emit();
            
            source2d.data['x0'][0] = new_xx[0];
            source2d.data['delta_x'][0] = new_xx[new_xx.length-1] - new_xx[0];
            source2d.data['g0'][0] = new_gg[0];
            source2d.data['delta_g'][0] = new_gg[new_gg.length-1] - new_gg[0];
            source2d.change.emit();
            """
    )
    for slider in sliders_dict.values():
        slider.js_on_change('value', slider_callback)

    # HOVERING
    hover_source_dict = {
        'primalpoint': primalpoint.data_source,
        'primaltangent': primaltangent.data_source,
        'primaldroite': primaldroite.data_source,
        'primalheight': primalheight.data_source,
        'primalgap': primalgap.data_source,
        'dualpoint': dualpoint.data_source,
        'dualtangent': dualtangent.data_source,
        'dualdroite': dualdroite.data_source,
        'dualheight': dualheight.data_source,
        'dualgap': dualgap.data_source,
        'hpoint': hpoint.data_source,
        'vpoint': vpoint.data_source,
        'hline': hline.data_source,
        'vline': vline.data_source,
    }

    source_dict = {
        'primal': primal_source,
        'dual': dual_source,
        **hover_source_dict
    }

    # hover over primal plot
    primalhoverjs = CustomJS(
        args=source_dict,
        code="""
            let x0 = cb_obj.x;
            let y0 = cb_obj.y;
            
            let xx = primal.data['xx'];
            let i = xx.findIndex(x => x >=x0);
            if (i==-1){i = xx.length-1};
            let x1 = xx[i];
            let y1 = primal.data['fcc'][i];
            primalpoint.data['x'] = [x1];
            primalpoint.data['y'] = [y1];
            primalpoint.change.emit();
            
            primalheight.data['x'] = [x1,x1];
            primalheight.data['y'] = [y0,y1]
            primalheight.change.emit();
    
            let j = primal.data['idgopt'][i];
            let gg = dual.data['gg'];
            let g1 = gg[j];
            let fc1 = dual.data['fc'][j];        
            dualpoint.data['g'] = [g1];
            dualpoint.data['y'] = [fc1];
            dualpoint.change.emit();
    
            dualheight.data['g'] = [g1,g1];
            dualheight.data['y'] = [x0*g1 - y0, fc1];
            dualheight.change.emit();
            
            dualtangent.data['g'] = [gg[0]-1000, gg[gg.length-1]+1000];
            dualtangent.data['y'] = dualtangent.data['g'].map(g => fc1 + x1*(g-g1));
            dualtangent.change.emit();
    
            dualdroite.data.g = dualtangent.data.g;
            dualdroite.data.y = dualdroite.data.g.map(g => x0*g - y0);
            dualdroite.change.emit();
            
            vpoint.data['x'] = [x1];
            vpoint.data['g'] = [g1];
            vpoint.change.emit();   
                 
            vline.data['x'] = [x1, x1];
            vline.data['g'] = [gg[0], gg[gg.length - 1]];
            vline.change.emit();
            """
    )
    fig1.js_on_event(bokeh.events.MouseMove, primalhoverjs)
    fig1.js_on_event(bokeh.events.Tap, primalhoverjs)

    # hover over dual plot
    dualhoverjs = CustomJS(
        args=source_dict,
        code="""
            let g0 = cb_obj.x;
            let y0 = cb_obj.y;
            
            let gg = dual.data['gg'];
            let j = gg.findIndex(g => g >=g0);
            if (j==-1){j = gg.length-1};
            let g1 = gg[j];
            let fc1 = dual.data['fc'][j];
            dualpoint.data['g'] = [g1];
            dualpoint.data['y'] = [fc1];
            dualpoint.change.emit();
    
            dualheight.data['g'] = [g1, g1];
            dualheight.data['y'] = [y0, fc1];
            dualheight.change.emit();
    
            let i = dual.data['idxopt'][j];
            let xx = primal.data['xx'];
            let x1 = xx[i];
            let ff1 = primal.data['ff'][i];        
            primalpoint.data['x'] = [x1];
            primalpoint.data['y'] = [ff1];
            primalpoint.change.emit();
    
            primaltangent.data['x'] = [xx[0]-1000, xx[xx.length-1]+1000];
            primaltangent.data['y'] = primaltangent.data['x'].map(x => g0*(x-x1) + ff1);
            primaltangent.change.emit();
            
            primaldroite.data.x = [xx[0]-1000, xx[xx.length-1]+1000];
            primaldroite.data.y = primaldroite.data.x.map(x => g0 * x - y0);
            primaldroite.change.emit();
    
            primalheight.data['x'] = [x1, x1];
            primalheight.data['y'] = [g0*x1 - y0, ff1];
            primalheight.change.emit();
    
            hpoint.data['x'] = [x1];
            hpoint.data['g'] = [g1];
            hpoint.change.emit();        
            
            hline.data['x'] = [xx[0], xx[xx.length - 1]];
            hline.data['g'] = [g1, g1];
            hline.change.emit();
            """
    )
    fig2.js_on_event(bokeh.events.MouseMove, dualhoverjs)
    fig2.js_on_event(bokeh.events.Tap, dualhoverjs)

    # Hover over X,G plots
    code_get_xg = """
        let x0 = cb_obj.x;
        let g0 = cb_obj.y;
        let xx = primal.data['xx'];
        let gg = dual.data['gg'];
        
        let i = xx.findIndex(x => x >=x0);
        if (i==-1){i = xx.length-1};
        let x1 = xx[i];
        let ff1 = primal.data['ff'][i];
        primalpoint.data['x'] = [x1];
        primalpoint.data['y'] = [ff1];
        primalpoint.change.emit();
        
        let j = gg.findIndex(g => g >=g0);
        if (j==-1){j = gg.length-1};
        let g1 = gg[j];
        let fc1 = dual.data['fc'][j];
        dualpoint.data['g'] = [g1];
        dualpoint.data['y'] = [fc1];
        dualpoint.change.emit();
        """

    hover_events = [bokeh.events.MouseMove, bokeh.events.Tap]

    # hover over g.x-f(x)
    for event in hover_events:
        images[0].js_on_event(event, CustomJS(
            args=source_dict,
            code=code_get_xg + """
            hpoint.data['x'] = [x1];
            hpoint.data['g'] = [g1];
            hpoint.change.emit();        
    
            hline.data['x'] = [xx[0], xx[xx.length - 1]];
            hline.data['g'] = [g1, g1];
            hline.change.emit();
            
            primaltangent.data['x'] = [xx[0]-1000, xx[xx.length-1]+1000];
            primaltangent.data['y'] = primaltangent.data['x'].map(x => g1*x);
            primaltangent.change.emit();
    
            primalheight.data['x'] = [x1,x1];
            primalheight.data['y'] = [g1*x1, ff1];
            primalheight.change.emit();
            
            dualheight.data['g'] = [g1, g1];
            dualheight.data['y'] = [0, g1*x1-ff1];
            dualheight.change.emit();
            
            dualgap.data['g'] = [g1, g1];
            dualgap.data['y'] = [g1*x1-ff1, fc1];
            dualgap.change.emit();
            """
        ))

    # hover over g.x-f*(g)
    for event in hover_events:
        images[1].js_on_event(event, CustomJS(
            args=source_dict,
            code=code_get_xg + """
            vpoint.data['x'] = [x1];
            vpoint.data['g'] = [g1];
            vpoint.change.emit();   
                 
            vline.data['x'] = [x1, x1];
            vline.data['g'] = [gg[0], gg[gg.length - 1]];
            vline.change.emit();
            
            dualtangent.data['g'] = [gg[0]-1000, gg[gg.length-1]+1000];
            dualtangent.data['y'] = dualtangent.data['g'].map(g => x1*g);
            dualtangent.change.emit();
            
            dualheight.data['g'] = [g1,g1];
            dualheight.data['y'] = [fc1, g1*x1];
            dualheight.change.emit();
    
            primalheight.data['x'] = [x1, x1];
            primalheight.data['y'] = [0, g1*x1-fc1];
            primalheight.change.emit();
            
            primalgap.data['x'] = [x1, x1];
            primalgap.data['y'] = [g1*x1-fc1, ff1];
            primalgap.change.emit();
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

    bigfig = layouts.gridplot([
        [fig1, fig2, fig3],
        images,
        [sliders_dict['x_shift'], sliders_dict['f_shift'], sliders_dict['g_shift']],
        [sliders_dict['x_dilate'], sliders_dict['f_dilate'], None],
        # [sliders_dict['x_shift'], sliders_dict['f_shift'], None],
    ], toolbar_location=None)
    return bigfig
