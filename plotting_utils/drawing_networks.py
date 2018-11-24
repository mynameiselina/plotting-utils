# IMPORTS
import sys
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, ListedColormap
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from .core import (
    custom_div_cmap, MyColors
)
import logging
logger = logging.getLogger(__name__)


def networkx_init_graph(
        similarity, n_nodes, color='weight', orderBy='abs_weight'):
    # CREATE graph object
    if similarity is None:
        logger.info('generating random graph...')
        G = nx.newman_watts_strogatz_graph(n_nodes, 4, 0.15, seed=0)

        n_edges = G.number_of_edges()
        logger.info('number of edges to draw: '+str(n_edges))

        e1, e2 = zip(*[(e[0], e[1]) for e in G.edges(data=True)])
        e1_ar = np.array(e1)
        e2_ar = np.array(e2)

        # create also edge attributes
        edge_color = np.repeat(1, n_edges)
        edge_order = np.arange(n_edges)
        # The position of name and values has been swapped from 1.x to 2.0
        nx.set_edge_attributes(G, name='weight', values=edge_color)
        nx.set_edge_attributes(G, name='abs_weight', values=edge_order)
        del e1, e2

    else:
        if len(similarity.shape) == 1:
            logger.info('the similarity data format is: condensed')
            logger.warning('REVISE CODE to correct the graph nodes order!')

            G = nx.Graph()

            # create  edge list (of tuples) to create graph
            e1_, e2_ = np.triu_indices(n_nodes, 1)
            nnz = similarity.nonzero()[0]
            egde_list = list(zip(e1_[nnz], e2_[nnz]))

            n_edges = len(egde_list)
            logger.info('number of edges to draw: '+str(n_edges))
            if n_edges > 0:
                # INIT the graph with the weighted edges
                G.add_edges_from(egde_list)
                del egde_list

                # create also edge attributes
                weights = dict(
                    zip(list(zip(e1_[nnz], e2_[nnz])), similarity[nnz]))
                abs_weights = dict(
                    zip(list(zip(e1_[nnz], e2_[nnz])), abs(similarity[nnz])))
                # The position of name and values was swapped from 1.x to 2.0
                nx.set_edge_attributes(G, name='weight', values=weights)
                nx.set_edge_attributes(
                    G, name='abs_weight', values=abs_weights)
                del e1_, e2_, nnz, weights, abs_weights

                e1, e2, c, ep = zip(*[
                    (e[0], e[1], e[2][color], e[2][orderBy])
                    for e in G.edges(data=True)])
                e1_ar = np.array(e1)
                e2_ar = np.array(e2)
                edge_color = np.array(c)
                edge_order = abs(np.array(ep)).argsort()
                del e1, e2, c, ep
            else:
                e1_ar = []
                e2_ar = []
                edge_color = []
                edge_order = []

            if len(G.nodes()) < n_nodes:
                logger.info('adding nodes...')
                G.add_nodes_from(
                    list(set(np.arange(n_nodes)) - set(G.nodes())))
        else:
            if sp.sparse.issparse(similarity):
                logger.info(
                    'the similarity data format is: ' +
                    similarity.getformat())
                G = nx.from_scipy_sparse_matrix(similarity)
                logger.info(
                    'number of edges to draw: ' +
                    str(G.number_of_edges()))

                # e1, e2, c, ep = zip(*[(e[0], e[1], e[2], abs(e[2]))
                # for e in list(G.edges_iter(data='weight'))]) # v1.x
                e1, e2, c, ep = zip(*[
                    (e[0], e[1], e[2], abs(e[2]))
                    for e in list(G.edges(data='weight'))])
                abs_weights = dict(zip(list(zip(e1, e2)), ep))
                nx.set_edge_attributes(
                    G, name='abs_weight', values=abs_weights)
                e1_ar = np.array(e1)
                e2_ar = np.array(e2)
                edge_color = np.array(c)
                edge_order = abs(np.array(ep)).argsort()

                del e1, e2, c, ep

            else:
                logger.error(
                    'cannot recognize the similarity data format! ' +
                    'Acceptable formats are sparse or condensed.')
                raise

    return G, e1_ar, e2_ar, edge_color, edge_order


def networkx_set_edges(
        G, edge_color, edge_order, e1, e2, which_nodes=None,
        all_edges=False, edge_color_min=None, edge_color_max=None,
        edgeCmap='default', palette="gray_r"):
    # ---- EDGES ---- #
    # set the colormap for the edges according to the weights (not abs here!)
    if edge_color_min is None:
        edge_color_min = edge_color.min()
    if edge_color_max is None:
        edge_color_max = edge_color.max()
    clim_edge = [edge_color_min, edge_color_max]

    if edgeCmap == "default":
        if clim_edge[0] == clim_edge[1]:
            cmap_edge = custom_div_cmap(
                1, mincol='#000000', midcol='#000000', maxcol='#000000')
        else:
            if clim_edge[0] < 0:
                # if we have pos and neg values
                # but the zero is not in the midle of the range then adjust
                newLim = abs(np.array(clim_edge)).max()
                clim_edge = [-newLim, newLim]
                cmap_edge = custom_div_cmap(
                    1000, mincol='blue', midcol='white', maxcol='red')
            else:
                # #f0f8ff Color Hex Alice Blue
                # #dcdcdc Color Hex Gainsboro (~gray)
                cmap_edge = custom_div_cmap(
                    1000, mincol='#F0F8FF', midcol='#DCDCDC', maxcol='#000000')

    elif edgeCmap == 'cat_n':
        uniq_colors = np.unique(edge_color)
        n = len(uniq_colors)
        mycolor_palette = sns.color_palette(palette, n)
        mycolor_palette_hex = mycolor_palette.as_hex()
        cmap_edge = ListedColormap(mycolor_palette_hex)

    elif edgeCmap == 'cat_cont':
        n = 1000
        mycolor_palette = sns.color_palette(palette, n)
        mycolor_palette_hex = mycolor_palette.as_hex()
        cmap_edge = ListedColormap(mycolor_palette_hex)

    # map a color for the edges after you Normalize the color values
    alpha = abs(edge_color) / abs(edge_color).max()  # transparency
    norm_edge = Normalize(vmin=clim_edge[0], vmax=clim_edge[1])
    edge_norm = norm_edge(edge_color)
    edge_color = cmap_edge(edge_norm)
    edge_color[:, 3] = alpha
    edge_color[(edge_color[:, 3] == 0), 3] = 0.0001
    del alpha, clim_edge

    edge_width = edge_norm

    # if we are drawing a subgraph
    if which_nodes is None:
        # np.array(G.edges(data=True))[edge_order].tolist() # v1.x
        edges2draw = np.array(list(G.edges(data=True)))[edge_order].tolist()
        edge_colors2draw = edge_color[edge_order]
        edge_widths2draw = edge_width[edge_order]

        edgesNOT2draw = None
        edge_widthsNOT2draw = None
        edge_colorsNOT2draw = None
    else:
        if all_edges:
            which_edges = np.in1d(e1[edge_order], which_nodes) | \
                np.in1d(e2[edge_order], which_nodes)
            # v1.x
            # # np.array(G.edges(data=True))[edge_order][ which_edges].tolist()
            edges2draw = np.array(
                list(G.edges(data=True)))[edge_order][which_edges].tolist()

            n1, n2 = zip(*[(e[0], e[1]) for e in edges2draw])
            which_nodes = list(set(n1).union(set(n2)))
        else:
            which_edges = np.in1d(e1[edge_order], which_nodes) & \
                np.in1d(e2[edge_order], which_nodes)
            # v1.x
            # # np.array(G.edges(data=True))[edge_order][ which_edges].tolist()
            edges2draw = np.array(
                list(G.edges(data=True)))[edge_order][which_edges].tolist()

        # v1.x
        # # np.array(G.edges(data=True))[edge_order][~which_edges].tolist()
        edgesNOT2draw = np.array(
            list(G.edges(data=True)))[edge_order][~which_edges].tolist()
        edge_colors2draw = edge_color[edge_order][which_edges]
        edge_widths2draw = edge_width[edge_order][which_edges]

        edge_widthsNOT2draw = edge_width[edge_order][~which_edges]
        edge_colorsNOT2draw = edge_color[edge_order][~which_edges]

        del which_edges

    return cmap_edge, norm_edge, edges2draw, edge_colors2draw, \
        edge_widths2draw, edgesNOT2draw, edge_colorsNOT2draw, \
        edge_widthsNOT2draw, which_nodes


def networkx_set_nodes(
        G, n_nodes, node_colors, node_sizes, size_const, which_nodes=None,
        cmap='default', palette="tab10", nodeColor_min=None,
        nodeColor_max=None):
    # ---- NODES ---- #
    # colormap for nodes
    if nodeColor_max is None:
        nodeColor_max = node_colors.max()
    if nodeColor_min is None:
        nodeColor_min = node_colors.min()

    if cmap == 'default':
        cmap_node = custom_div_cmap(
            1000, mincol='yellow', midcol='orange', maxcol='darkred')
    elif cmap == 'btwn':
        cmap_node = custom_div_cmap(
            1000, mincol='gray', midcol='lightblue', maxcol='darkblue')
    elif cmap == 'chr':
        nodeColor_max = max(node_colors.max(), 23)
        nodeColor_min = 1
        if palette is None:
            myCol_obj = MyColors()
            myExtra = myCol_obj.get_colors(order=[
                125, 5, 112, 19, 46, 123, 78, 120, 80, 87, 89, 90,
                33, 26, 145, 65, 93, 140, 76, 127, 133, 52, 59])
            myExtra = myExtra[nodeColor_min-1:nodeColor_max]
            mycolor_palette = sns.color_palette(myExtra)
        else:
            mycolor_palette = sns.color_palette(palette)
        cmap_node = ListedColormap(mycolor_palette.as_hex())
    elif cmap == 'cat':
        uniq_colors = np.unique(node_colors)
        n = node_colors.max()
        if 0 in uniq_colors:
            n += 1
        mycolor_palette = sns.color_palette(palette, n)
        mycolor_palette_hex = mycolor_palette.as_hex()
        cmap_node = ListedColormap(mycolor_palette_hex)
    elif cmap == 'cat_n':
        if nodeColor_max is None:
            n = node_colors.max()
        else:
            n = nodeColor_max
        if 0 in [node_colors.min(), nodeColor_min]:
            n += 1
        mycolor_palette = sns.color_palette(palette, n)
        mycolor_palette_hex = mycolor_palette.as_hex()
        cmap_node = ListedColormap(mycolor_palette_hex)
    elif cmap == 'cat_cont':
        n = len(np.unique(node_colors))
        cmap_node = custom_div_cmap(
            n, mincol='yellow', midcol='orange', maxcol='darkred')

    # map a color for the nodes values after you Normalize the color values
    clim_node = [nodeColor_min, nodeColor_max]
    norm_node = Normalize(vmin=clim_node[0], vmax=clim_node[1])
    node_colors = cmap_node(norm_node(node_colors))

    # map a size for the atom nodes
    node_sizes = node_sizes*size_const

    # get the order of the sizes to draw the big first and the small after
    # (so that the appear on the front)
    node_order = node_sizes.argsort()[::-1]

    # if drawing a sub-graphs
    if which_nodes is None:
        nodes2draw = np.array(sorted(G.nodes()))[node_order].tolist()
        node_colors2draw = node_colors[node_order]
        node_sizes2draw = node_sizes[node_order]
        nodesNOT2draw = None
        node_sizesNOT2draw = None
        node_colorsNOT2draw = None

    else:
        which_nodes_bool = np.ones(n_nodes, bool)
        which_nodes_bool[which_nodes] = False
        which_nodes_bool = ~which_nodes_bool
        which_nodes_inorder = which_nodes_bool[node_order]

        nodes2draw = np.array(
            sorted(G.nodes()))[node_order][which_nodes_inorder].tolist()
        nodesNOT2draw = np.array(
            sorted(G.nodes()))[node_order][~which_nodes_inorder].tolist()

        node_colors2draw = node_colors[node_order][which_nodes_inorder]
        node_colorsNOT2draw = node_colors[node_order][~which_nodes_inorder]

        node_sizes2draw = node_sizes[node_order][which_nodes_inorder]
        node_sizesNOT2draw = node_sizes[node_order][~which_nodes_inorder]

    return clim_node, cmap_node, norm_node, nodes2draw, node_colors2draw, \
        node_sizes2draw, nodesNOT2draw, node_colorsNOT2draw, node_sizesNOT2draw


def networkx_set_pie_nodes(
        G, n_nodes, node_colors, node_sizes, size_const, which_nodes=None,
        cmap='default', palette="tab10", nodeColor_min=None,
        nodeColor_max=None):
    # ---- NODES ---- #
    # colormap for nodes
    if nodeColor_max is None:
        nodeColor_max = node_colors.shape[1]-1
    if nodeColor_min is None:
        nodeColor_min = 0

    n = node_colors.shape[1]

    if cmap == 'default':
        cmap_node = custom_div_cmap(
            1000, mincol='yellow', midcol='orange', maxcol='darkred')
    elif cmap == 'btwn':
        cmap_node = custom_div_cmap(
            1000, mincol='gray', midcol='lightblue', maxcol='darkblue')
    elif cmap == 'chr':
        myCol_obj = MyColors()
        myExtra = myCol_obj.get_colors(order=[
            45, 125, 5, 112, 19, 46, 123, 78, 120, 80, 87, 89, 90,
            33, 26, 145, 65, 93, 140, 76, 127, 133, 52, 59])
        myExtra = myExtra[n]
        mycolor_palette = sns.color_palette(myExtra)
        cmap_node = ListedColormap(mycolor_palette.as_hex())
    elif cmap == 'cat':
        mycolor_palette = sns.color_palette(palette, n)
        cmap_node = ListedColormap(mycolor_palette.as_hex())
    elif cmap == 'cat_cont':
        cmap_node = custom_div_cmap(
            n, mincol='yellow', midcol='orange', maxcol='darkred')

    # map a color for the nodes pie wedges,
    # after you Normalize the color values
    clim_node = [nodeColor_min, nodeColor_max]
    norm_node = Normalize(vmin=clim_node[0], vmax=clim_node[1])

    # map a size for the atom nodes
    if node_sizes.max() > 1:
        node_sizes = node_sizes/node_sizes.max()
    node_sizes = node_sizes*size_const
    # node_sizes_smaller =node_sizes/5

    # get the order of the sizes to draw the big first and the small after
    # (so that the appear on the front)
    node_order = node_sizes.argsort()[::-1]

    # if drawing a sub-graphs
    if which_nodes is None:

        nodes2draw = np.array(sorted(G.nodes()))[node_order].tolist()
        # node_colors2draw = node_colors[node_order, :]
        # node_sizes2draw = node_sizes[node_order]
        nodesNOT2draw = None
        # node_sizesNOT2draw = None
        # node_colorsNOT2draw = None

    else:
        which_nodes_bool = np.ones(n_nodes, bool)
        which_nodes_bool[which_nodes] = False
        which_nodes_bool = ~which_nodes_bool
        which_nodes_inorder = which_nodes_bool[node_order]

        nodes2draw = np.array(
            sorted(G.nodes()))[node_order][which_nodes_inorder].tolist()
        nodesNOT2draw = np.array(
            sorted(G.nodes()))[node_order][~which_nodes_inorder].tolist()

        # node_colors2draw    = node_colors[node_order,:][
        #   which_nodes_inorder, :]
        # node_colorsNOT2draw = node_colors[node_order,:][
        #   ~which_nodes_inorder, :]

        # node_sizes2draw    = node_sizes[node_order][ which_nodes_inorder]
        # node_sizesNOT2draw = node_sizes[node_order][~which_nodes_inorder]

    return clim_node, cmap_node, norm_node, nodes2draw, node_colors, \
        node_sizes, nodesNOT2draw, node_colors, node_sizes


def networkx_set_layout(
        G, layout, specified_pos, pos_param=None,
        weight='abs_weight', mySeed=8):
    # ---- LAYOUT ---- #
    np.random.seed(mySeed)
    if specified_pos is None:
        if layout == 'default':
            logger.info('default with weights = '+weight)
            pos = nx.spring_layout(G, weight=weight)
        elif layout == 'spring1':
            logger.info(
                'spring_layout with weights = ' +
                weight+' and k = '+str(pos_param))
            pos = nx.spring_layout(G, k=pos_param, weight=weight)
        elif layout == 'spring2':
            logger.info(
                'spring_layout with weights = '+weight +
                ', k = '+str(pos_param)+' and iterations = 200')
            pos = nx.spring_layout(
                G, k=pos_param, iterations=200, weight=weight)
        elif layout == 'circular':
            logger.info('circular_layout')
            pos = nx.circular_layout(G)
        elif layout == 'spectral':
            logger.info('spectral_layout')
            pos = nx.spectral_layout(G,  weight=weight)
        elif 'graphviz' in layout:
            logger.info('graphviz_layout')
            prog = layout.rsplit('__')[-1]
            logger.info(prog)
            pos = graphviz_layout(G, prog=prog)
        elif layout == 'none':
            logger.info('no layout')
            pos = None
        elif layout == 'kamada_kawai':
            logger.info('kamada_kawai with weights = '+weight)
            pos = nx.kamada_kawai_layout(G, weight=weight, scale=pos_param)
        else:
            logger.error('Invalid layout setting!'+weight)
            raise

    else:
        pos = specified_pos

    return pos


def networkx_draw(
        G, pos, myfigsize, which_nodes, mytitle, with_names,
        n_edges, edges2draw, edge_colors2draw, edge_widths2draw, edgesNOT2draw,
        edge_colorsNOT2draw, edge_widthsNOT2draw, edge_line_width,
        nodes2draw, nodesNOT2draw, node_colors2draw, node_sizes2draw,
        node_colorsNOT2draw, node_sizesNOT2draw, node_line_width,
        cmap_edge, norm_edge, cmap_node, norm_node, clim_node, nodeCmap,
        nodeCmap_ticklabels, print_node_names, print_edge_names, font_size,
        font_color, nodes_as_pies):

    _positions = pd.DataFrame.from_dict(pos)
    pos_min_lims = _positions.min(axis=1).values
    pos_max_lims = _positions.max(axis=1).values

    # ---- DRAW ---- #
    # init figure and subplots for colorbars
    myfig = plt.figure(figsize=myfigsize)

    # grid size
    r = 500
    c = 500
    s = 10
    ax1 = plt.subplot2grid((r, c), (0, 0), colspan=c-s, rowspan=r-s)
    ax2 = plt.subplot2grid((r, c), (0, c-s), colspan=s, rowspan=r-s)
    ax3 = plt.subplot2grid((r, c), (r-s, 0), colspan=c-s, rowspan=s)

    # set axis lims (because we will draw first the edges and then the nodes,
    # otherwise we don't need to set the lims)
    if nodes_as_pies:
        a_size = (pos_max_lims.max()-pos_min_lims.min())/10
        a2 = a_size/2.0
    else:
        a2 = 0.2
    ax1.set_xlim(pos_min_lims[0]-a2, pos_max_lims[0]+a2)
    ax1.set_ylim(pos_min_lims[1]-a2, pos_max_lims[1]+a2)

    # draw edges
    if n_edges > 0:
        # draw edges
        if edge_line_width < 0:
            if which_nodes is not None:
                edges = nx.draw_networkx_edges(
                    G, pos, edgelist=edgesNOT2draw,
                    width=edge_widthsNOT2draw*abs(edge_line_width),
                    edge_color=edge_colorsNOT2draw, alpha=0.25, ax=ax1)
            edges = nx.draw_networkx_edges(
                G, pos, edgelist=edges2draw,
                width=edge_widths2draw*abs(edge_line_width),
                edge_color=edge_colors2draw, ax=ax1)
        else:
            if which_nodes is not None:
                edges = nx.draw_networkx_edges(
                    G, pos, edgelist=edgesNOT2draw,
                    width=edge_line_width/2,
                    edge_color='grey', alpha=0.25, ax=ax1)
            edges = nx.draw_networkx_edges(
                G, pos, edgelist=edges2draw,
                width=edge_line_width,
                edge_color=edge_colors2draw, ax=ax1)

        if print_edge_names:
            edge_labels = {
                i[0:2]: '{}'.format(i[2]['weight']) for i in edges2draw}
            edge_label_handles = nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, ax=ax1)
            [
                label.set_bbox(dict(facecolor='none', edgecolor='none'))
                for label in edge_label_handles.values()
            ]

        # plot colorbar for edges in the right vertical axis
        bar = ColorbarBase(
            cmap=cmap_edge, norm=norm_edge, ax=ax2)
        bar.ax.tick_params(labelsize=myfigsize[0])
    else:
        ax2.axis('off')

    if nodes_as_pies:

        # draw nodes labels
        if print_node_names:
            if with_names is None:
                node_labels = {e: str(e) for e in sorted(G.nodes())}
            else:
                node_labels = {e: with_names[e] for e in sorted(nodes2draw)}
            # logger.info('node_labels'+str(node_labels))

        # transform from your data to your display coordinate system
        trans = ax1.transData.transform
        # inverted() transform from display to figure coordinate system
        trans2 = myfig.transFigure.inverted().transform

        wedge_colors = [cmap_node(i) for i in range(node_colors2draw.shape[1])]

        if which_nodes is not None:
            for n in nodesNOT2draw:
                node_s = node_sizesNOT2draw[n]

                xx, yy = trans(pos[n])  # figure coordinates
                xa, ya = trans2((xx, yy))  # axes coordinates
                a = plt.axes([xa-a2, ya-a2, a_size, a_size])
                # a.set_aspect('equal')
                node_c = node_colorsNOT2draw[n, :]
                if node_line_width > 0:
                    a.pie(
                        node_c, radius=node_s, colors=['lightgrey'],
                        wedgeprops={
                            'linewidth': node_line_width,
                            'edgecolor': 'lightgrey'})
                else:
                    a.pie(node_c, radius=node_s, colors=['lightgrey'])

        for n in nodes2draw:
            node_s = node_sizes2draw[n]

            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            a = plt.axes([xa-a2, ya-a2, a_size, a_size])
            # a.set_aspect('equal')
            node_c = node_colors2draw[n, :]
            if node_line_width > 0:
                wedgeprops = {
                    'linewidth': node_line_width,
                    'edgecolor': 'black'}
            else:
                wedgeprops = {}

            a.pie(
                node_c, radius=node_s,
                colors=wedge_colors, wedgeprops=wedgeprops)
            if print_node_names:
                # transform from your data to your display coordinate system
                # trans3=a.transData.transform
                # inverted() transform from display to axes coordinate system
                # trans4=a.transAxes.inverted().transform
                # logger.info('data coordinates '+str(xx,yy+node_s))
                # dx, dy = trans3((xx,yy+node_s))
                # logger.info('display coordinates '+str([dx, dy]))
                # tx, ty = trans4((dx, dy))
                # logger.info('axes coordinates '+str([tx, ty]))
                a.set_title(
                    node_labels[n], x=0.5,  y=0.5, fontdict={
                        'fontsize': font_size,
                        'fontweight': abs(edge_line_width),
                        'color': font_color,
                        'weight': 'bold',
                        'alpha': 0.8})
    else:
        # draw nodes
        if which_nodes is not None:
            nodes = nx.draw_networkx_nodes(
                G, pos, nodelist=nodesNOT2draw,
                node_color='lightgrey',
                node_size=node_sizesNOT2draw,
                linewidths=0.1, ax=ax1)
            if nodes is not None:
                nodes.set_edgecolor('grey')

        nodes = nx.draw_networkx_nodes(
            G, pos, nodelist=nodes2draw,
            node_color=node_colors2draw,
            node_size=node_sizes2draw,
            linewidths=node_line_width, ax=ax1)
        if node_line_width > 0:
            nodes.set_edgecolor('black')

        # draw nodes labels
        if print_node_names:
            if with_names is None:
                node_labels = {e: str(e) for e in sorted(G.nodes())}
                node_label_handles = nx.draw_networkx_labels(
                    G, pos, labels=node_labels,
                    font_size=font_size,
                    font_color=font_color,
                    font_weight='bold',
                    ax=ax1)
            else:
                node_labels = {
                    e: with_names[i] for i, e in enumerate(sorted(nodes2draw))}
                node_label_handles = nx.draw_networkx_labels(
                    G, pos, labels=node_labels,
                    font_size=font_size,
                    font_color=font_color,
                    font_weight='bold',
                    ax=ax1)

    # plot colorbar for nodes in the bottom horizontal axis
    if nodeCmap == 'btwn':
        bar = ColorbarBase(
            cmap=cmap_node, norm=norm_node,
            # ticks=np.arange(clim_node[0],clim_node[1]+1,1),
            # np.arange(clim_node[0],clim_node[1]+1,1)
            orientation='horizontal',
            ax=ax3)
    else:
        bar = ColorbarBase(
            cmap=cmap_node, norm=norm_node,
            ticks=np.arange(clim_node[0], clim_node[1]+1, 1),
            orientation='horizontal',
            ax=ax3)
    if nodeCmap_ticklabels is not None:
        bar.set_ticklabels(nodeCmap_ticklabels, update_ticks=True)
    bar.ax.tick_params(labelsize=myfigsize[0])

    ax1.axis('off')
    ax1.set_title(mytitle, fontsize=font_size)

    return myfig


def networkx_draw_network(
        similarity, node_sizes, node_colors, size_const=10,
        which_nodes=None, all_edges=False, print_edge_names=False,
        layout='spring', specified_pos=None, pos_param=None,
        myfigsize=(20, 20), mytitle='',
        print_node_names=True, font_size=15, with_names=None,
        font_color='black',
        edgeCmap='default', edgePalette="tab10",
        edge_color_min=None, edge_color_max=None, edge_line_width=3.5,
        nodeCmap='default', nodeCmap_ticklabels=None, nodePalette="tab10",
        nodeColor_min=None, nodeColor_max=None, node_line_width=0,
        nodes_as_pies=False,
        mySeed=8):

    n_nodes = len(node_colors)

    # ---- INIT GRAPH ---- #
    G, e1_ar, e2_ar, edge_color, edge_order = networkx_init_graph(
        similarity, n_nodes, color='weight', orderBy='abs_weight')
    n_edges = len(G.edges())

    # ---- EDGES ---- #
    if n_edges > 0:
        cmap_edge, norm_edge, edges2draw, edge_colors2draw, edge_widths2draw, \
            edgesNOT2draw, edge_colorsNOT2draw, edge_widthsNOT2draw, \
            which_nodes = networkx_set_edges(
                G, edge_color, edge_order, e1_ar, e2_ar,
                which_nodes=which_nodes, all_edges=all_edges,
                edge_color_min=edge_color_min,
                edge_color_max=edge_color_max, edgeCmap=edgeCmap,
                palette=edgePalette)
        del e1_ar, e2_ar
    else:
        cmap_edge = None
        norm_edge = None
        edges2draw = None
        edge_colors2draw = None
        edge_widths2draw = None
        edgesNOT2draw = None
        edge_colorsNOT2draw = None
        edge_widthsNOT2draw = None

    # ---- NODES ---- #
    if nodes_as_pies:
        clim_node, cmap_node, norm_node, nodes2draw, node_colors2draw, \
            node_sizes2draw, nodesNOT2draw, node_colorsNOT2draw, \
            node_sizesNOT2draw = networkx_set_pie_nodes(
                G, n_nodes, node_colors, node_sizes, size_const,
                which_nodes=which_nodes, cmap=nodeCmap, palette=nodePalette,
                nodeColor_min=nodeColor_min, nodeColor_max=nodeColor_max)
    else:
        clim_node, cmap_node, norm_node, nodes2draw, node_colors2draw, \
            node_sizes2draw, nodesNOT2draw, node_colorsNOT2draw, \
            node_sizesNOT2draw = networkx_set_nodes(
                G, n_nodes, node_colors, node_sizes, size_const,
                which_nodes=which_nodes, cmap=nodeCmap, palette=nodePalette,
                nodeColor_min=nodeColor_min, nodeColor_max=nodeColor_max)

    # ---- LAYOUT ---- #
    pos = networkx_set_layout(
        G, layout, specified_pos, pos_param,
        weight='abs_weight', mySeed=mySeed)

    # ---- DRAW ---- #
    myfig = networkx_draw(
        G, pos, myfigsize, which_nodes, mytitle, with_names,
        n_edges, edges2draw, edge_colors2draw, edge_widths2draw, edgesNOT2draw,
        edge_colorsNOT2draw, edge_widthsNOT2draw, edge_line_width,
        nodes2draw, nodesNOT2draw, node_colors2draw, node_sizes2draw,
        node_colorsNOT2draw, node_sizesNOT2draw, node_line_width,
        cmap_edge, norm_edge, cmap_node, norm_node, clim_node,
        nodeCmap, nodeCmap_ticklabels, print_node_names,
        print_edge_names, font_size, font_color, nodes_as_pies)

    return myfig, pos, G, which_nodes
