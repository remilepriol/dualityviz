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

def plot_conjugate(resolution=100, pixelsize=350):
    #########
    # SOURCES
    #########
    # a source is made of columns with equal lengths
    # all the primal and dual ingredients go to their respective source
    primal_source = ColumnDataSource(data=dict(
        xx=[], ff=[], grad=[], fcc=[],
        idgopt=[], gopt=[], gzeros=[]
    ))
    # note to self : gopt is also the gradient of f**
    dual_source = ColumnDataSource(data=dict(
        gg=[], fc=[], idxopt=[],
        xopt=[], xzeros=[]
    ))
    # for images it's a bit more complex.
    # each column has length 1.
    # The image itself is an element
    # along with coordinates and width
    images_dict = {
        'gxminusf': [],
        'gxminusfc': [],
        'youngs': []
    }
    source2d = ColumnDataSource(data={
        **images_dict,
        'x0': [], 'delta_x': [],
        'g0': [], 'delta_g': []
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
    fig1.line('xx', 'fcc', source=primal_source, line_width=3, color=primalcolor, alpha=.5)
    fig1.line('xx', 'ff', source=primal_source, line_width=3, color=primalcolor)

    # temporary hovering glyphs
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

    # plot the dual function
    fig2 = plotting.figure(title='Dual f*(g)', **opts,
                           tools='pan', x_axis_label='g', y_axis_label='y')
    fig2.line('gg', 'fc', source=dual_source, line_width=3, color=dualcolor)

    # temporary hovering glyphs
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

    # INPUT FUNCTION
    # complicated without going full blown JS
    inputfunc = bokeh.models.TextInput(title='f(x) =', value='x^4', name='functionInputBox')
    lower_x = bokeh.models.TextInput(title='minimum x =', value='-1', name='lowerXInput')
    upper_x = bokeh.models.TextInput(title='maximum x =', value='1', name='upperXInput')
    textinput_js = CustomJS(
        name='functionRefreshScript',
        args=dict(inputfunc=inputfunc, lower_x=lower_x, upper_x=upper_x, resolution=resolution,
                  primal_source=primal_source, dual_source=dual_source, source2d=source2d),
        code="""  
        let xmax = +lower_x.value;
        let xmin = +upper_x.value;
        
        // clean all arrays
        for (let sourceData of [primal_source.data, dual_source.data]){
            for (let name in sourceData){
                sourceData[name] = [];    
            }        
        }
        
        function linspace(min,max,n) {
            let step=(max-min)/(n-1);
            let out = [];
            for (let i=0; i<n; i++) {
                out.push(min + i * step);
            }
            return out;
        }
         
        let xx = primal_source.data.xx = linspace(xmin,xmax,resolution);
        
        function primalFunc(x) {
            let out;
            try {
                out = math.evaluate(inputfunc.value, {'x':x});
            } catch (e) {
                return  x;
            }
            if (out == undefined) return x;
            return out;   
        }
        let ff = primal_source.data.ff = xx.map(primalFunc);
        
        let inputDerivative = math.derivative(inputfunc.value, 'x')
        function primalDerivative(x) {
            let out;
            try {
                out = inputDerivative.evaluate({'x': x});
            } catch (e) {
                return  x;
            }
            if (out == undefined) return x;
            return out; 
        }
        let grad = primal_source.data.grad = xx.map(primalDerivative);
        
        let gmax = Math.max(...grad);
        let gmin = Math.min(...grad);
        let gg =  linspace(gmin,gmax,resolution);
        
        function customMax(array) {
            // return [max, argmax] 
            return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r));
        }
        
        function getConjugate(xx,gg,ff){
            let gxminusf = []; 
            let d = {gg:gg, fc:[], idxopt:[], xopt:[], xzeros:[]};
            for (let i=0; i<resolution; i++){
                let rowi = [];
                for(let j=0; j<resolution; j++){
                    rowi.push(gg[i]*xx[j] - ff[j])
                }
                gxminusf.push(rowi);
                let [max, argmax] = customMax(rowi);
                d.fc.push(max);
                d.idxopt.push(argmax)
                d.xopt.push(xx[argmax]);
                d.xzeros.push(0);
            }
            return [gxminusf, d];
        }
        
        let [gxminusf, dualdata] = getConjugate(xx,gg,ff);
        dual_source.data = dualdata;
        
        dual_source.change.emit();
        
        let [gxminusfc, envelopedata] = getConjugate(gg, xx, dual_source.data.fc)
        
        primal_source.data.fcc = envelopedata.fc;
        primal_source.data.idgopt = envelopedata.idxopt;
        primal_source.data.gopt = envelopedata.xopt;
        primal_source.data.gzeros = envelopedata.xzeros;
    
        primal_source.change.emit();
        """)

    for textinput in [inputfunc, lower_x, upper_x]:
        textinput.js_on_change('value', textinput_js)

    # RADIO BUTTON to select function
    function_samples = {
        'x^4': ('x^4', -1, 1),
        'x^4 + 4 x^3': ('x^4 + 4 x^3', -1, 1),
        'absolute value': ('max(x, -x)', -1, 1),
        'quadratic': ('1/2 x^2', -1, 1),
        'x^2 - cos(2 x)': ('x^2 - cos(2 x)', -5, 5),
        'sin(x)': ('sin(x)', -3.14, 0),
        'x sin(1/x)': ('x sin(1/x)', -1, 1),
        'Entropy': ('x log(x) + (1-x) log(1-x)', 0, 1),
        'Hinge': ('max(0, 1-x)', -2, 2),
        'Squared Hinge': ('max(0, 1-x)^2', -2, 2),
    }
    radio_function_selection = models.RadioButtonGroup(
        labels=list(function_samples.keys()), active=0
    )
    radio_function_selection.js_on_change('active', CustomJS(
        args=dict(function_samples=function_samples,
                  inputfunc=inputfunc, lower_x=lower_x, upper_x=upper_x,
                  textinput_js=textinput_js),
        code="""
        let active_button = cb_obj.active;
        console.log(active_button);
        [inputfunc.value, lower_x.value, upper_x.value] = function_samples[cb_obj.labels[
        active_button]].map(String);
        textinput_js.execute();
        """
    ))

    # HOVERING
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

    bigfig = layouts.column([
        layouts.row([inputfunc, lower_x, upper_x]),
        radio_function_selection,
        layouts.gridplot([
            [fig1, fig2, fig3],
            # images,
        ], toolbar_location=None)
    ])
    return bigfig
