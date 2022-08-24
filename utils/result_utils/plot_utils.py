# Modified version of the plotNet in the sumolib python package
#https://github.com/eclipse/sumo/blob/4f8605bda9efe8d13a2fc6fa78c47385412c1a5b/tools/sumolib/visualization/helpers.py
def plotNet(net, colors, widths, options, incident_edge_obj, ax, margin=1600, area=None):
    shapes = []
    c = []
    w = []
    for e in net._edges:
        shapes.append(e.getShape())
        if e._id in colors:
            c.append(colors[str(e._id)])
        else:
            c.append(options.defaultColor)
        if e._id in widths:
            w.append(widths[str(e._id)])
        else:
            w.append(options.defaultWidth)

    line_segments = LineCollection(shapes, linewidths=w, colors=c)
    ax = plt.gca()
    ax.add_collection(line_segments)
    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)
    
    (xmin, ymin, xmax, ymax) = incident_edge_obj.getBoundingBox()
    if area is not None:
        ax.set_xlim(area['xmin'] ,area['xmax'])
        ax.set_ylim(area['ymin'] ,area['ymax'])
    else:
        ax.set_xlim(xmin - margin ,xmax + margin)
        ax.set_ylim(ymin - margin ,ymax + margin)
    
    #ax.autoscale_view(True, True, True)
    return ax
