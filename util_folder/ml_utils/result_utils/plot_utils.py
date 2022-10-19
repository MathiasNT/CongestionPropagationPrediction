from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from utils.dotdict import DotDict
 
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

    if incident_edge_obj is not None: 
        (xmin, ymin, xmax, ymax) = incident_edge_obj.getBoundingBox()
    if area is not None:
        ax.set_xlim(area['xmin'] ,area['xmax'])
        ax.set_ylim(area['ymin'] ,area['ymax'])
    elif incident_edge_obj is not None:
        ax.set_xlim(xmin - margin ,xmax + margin)
        ax.set_ylim(ymin - margin ,ymax + margin)
    else: 
        ax.autoscale_view(True, True, True)
    return ax

def plotNet_colormap(net, array, widths, options, incident_edge_obj, ax, margin=1600, area=None):
    shapes = []
    w = []
    for e in net._edges:
        shapes.append(e.getShape())
        if e._id in widths:
            w.append(widths[str(e._id)])
        else:
            w.append(options.defaultWidth)

    line_segments = LineCollection(shapes, linewidths=w, array=array, cmap=plt.cm.viridis)
    ax = plt.gca()
    ax.add_collection(line_segments)
    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)

    if incident_edge_obj is not None: 
        (xmin, ymin, xmax, ymax) = incident_edge_obj.getBoundingBox()
    if area is not None:
        ax.set_xlim(area['xmin'] ,area['xmax'])
        ax.set_ylim(area['ymin'] ,area['ymax'])
    elif incident_edge_obj is not None:
        ax.set_xlim(xmin - margin ,xmax + margin)
        ax.set_ylim(ymin - margin ,ymax + margin)
    else: 
        ax.autoscale_view(True, True, True)
    return ax


def plot_classification_errors(y_hat, y_true, seq_num, incident_info, ind_to_edge, net, ax):

    ie_idx = incident_info[...,0][seq_num].int().numpy()
    ie_id = ind_to_edge[str(ie_idx)]

    pred_classes = y_hat[...,0] > 0
    true_classes = y_true[...,0]
    true_pos = pred_classes * true_classes
    false_pos = pred_classes.int() - true_pos.int()
    false_neg = true_classes.int() - true_pos.int()

    true_pos_idx = torch.where(true_pos[seq_num])[0].numpy()
    false_pos_idx = torch.where(false_pos[seq_num])[0].numpy()
    false_neg_idx = torch.where(false_neg[seq_num])[0].numpy()
    true_pos_id = [ind_to_edge[str(idx)] for idx in true_pos_idx]
    false_pos_id = [ind_to_edge[str(idx)] for idx in false_pos_idx]
    false_neg_id = [ind_to_edge[str(idx)] for idx in false_neg_idx]

    edge_colors = {}

    for id in true_pos_id:
        edge_colors[id] = 'green'
    for id in false_pos_id:
        edge_colors[id] = 'orange'
    for id in false_neg_id:
        edge_colors[id] = 'yellow'

    widths = {ie_id:10}
        
    plot_options = {'defaultColor': 'black',
                    'defaultWidth': 2}
    plot_options = DotDict(plot_options)
    plotNet(net, colors=edge_colors, widths=widths, options=plot_options, incident_edge_obj=None, ax=ax)


    tp_patch = mpatches.Patch(color='green', label='True positives')
    fp_patch = mpatches.Patch(color='orange', label='False positives')
    fn_patch = mpatches.Patch(color='yellow', label='False negatives')
    ax.legend(handles=[tp_patch, fp_patch, fn_patch])
    return ax


def plot_net_w_logits(y_hat, seq_num, incident_info, ind_to_edge, net, ax):

    ie_idx = incident_info[...,0][seq_num].int().numpy()
    ie_id = ind_to_edge[str(ie_idx)]

    edge_colors = {}
    logits = y_hat[seq_num][...,0]

    widths = {ie_id:20}
        
    plot_options = {'defaultColor': 'black',
                    'defaultWidth': 2}
    plot_options = DotDict(plot_options)
    plotNet_colormap(net, array=logits, widths=widths, options=plot_options, incident_edge_obj=None, ax=ax)
    return ax
