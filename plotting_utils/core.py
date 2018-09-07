import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist, squareform
from math import ceil
import logging
logger = logging.getLogger(__name__)


def plot_stacked_barplot(
        df, xticklabels=None, xticksxpos=None, title="stacked bar plot",
        savePlot=False, fpath='DATA/tempFig.png', myfigsize=(30, 5)):
    data2plot = df.cumsum(axis=1)
    N = data2plot.shape[1]

    colormap = sns.color_palette("tab10", N)
    fig = plt.figure(figsize=myfigsize)
    for i, color in zip(np.arange(N-1, -1, -1), colormap):
        plt.bar(
            np.arange(data2plot.shape[0]), data2plot.values[:, i],
            color=color, label=i, width=1)
    l2 = plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., ncol=1)

    plt.xticks(range(0, data2plot.shape[0], 200))

    if xticklabels is None:
        plt.xticks(range(0, df.shape[0], 1), df.index, rotation=90)
    elif xticklabels is False:
        plt.xticks([], rotation=90)
    else:
        if xticksxpos is not None:
            plt.xticks(xticksxpos, xticklabels, rotation=0)
        else:
            plt.xticks(range(0, df.shape[0], 1), xticklabels, rotation=90)
    plt.title(title)

    if savePlot:
        fig.savefig(fpath, bbox_extra_artists=(l2,), bbox_inches='tight')

    plt.show()


def plot_clustered_stacked(
        df, labels=None, xticklabels=None, xticksxpos=None, width=0.5,
        fontsize=12, myfigsize=(25, 5), title="multiple stacked bar plot",
        H="/", savePlot=False, fpath='DATA/tempFig.png', **kwargs):
    # source:
    # https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
    # Given a list of dataframes, with identical columns and index,
    # create a clustered stacked bar plot.
    # labels: is a list of the names of the dataframe, used for the legend
    # title: is a string for the title of the plot
    # H: is the hatch used for identification of the different dataframe

    # fig = plt.figure(figsize=myfigsize)

    n_col = len(df.columns)
    n_ind = len(df.index)

    fig, axe = plt.subplots(1, figsize=myfigsize)

    mycolor_palette = sns.color_palette("tab10", n_col)
    cmap = ListedColormap(mycolor_palette.as_hex())

    # width=1.0/(n_col)
    axe = df.plot(kind="bar",
                  linewidth=0,
                  stacked=True,
                  ax=axe,
                  legend=False,
                  grid=False,
                  colormap=cmap,
                  fontsize=fontsize,
                  width=1,
                  xticks=None,
                  **kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify

    if xticklabels is None:
        axe.set_xticklabels(df.index, rotation=90)
    elif xticklabels is False:
        axe.set_xticklabels([], rotation=90)
    else:
        if xticksxpos is not None:
            axe.set_xticks(xticksxpos)
            axe.set_xticklabels(xticklabels, rotation=0)
        else:
            axe.set_xticklabels(xticklabels, rotation=90)
    axe.set_title(title)

    # Add invisible data to add another legend
    n = []
    n.append(axe.bar(0, 0, color="gray", hatch=0))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    else:
        l2 = plt.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])

    # axe.add_artist(l1)

    if savePlot:
        fig.savefig(fpath, bbox_extra_artists=(l2,), bbox_inches='tight')

# def plot_clustered_stacked(
#       dfall, labels=None, xticklabels=None, xticksxpos=None,
#       fontsize=12, myfigsize=(25,5), title="multiple stacked bar plot",
#       H="/", savePlot=False, fpath='DATA/tempFig.png', **kwargs):
# 	# source:
# https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
# 	# Given a list of dataframes, with identical columns and index,
#   create a clustered stacked bar plot.
# 	# labels: is a list of the names of the dataframe, used for the legend
# 	# title: is a string for the title of the plot
# 	# H: is the hatch used for identification of the different dataframe
#
# 	# fig = plt.figure(figsize=myfigsize)
#
# 	n_df = len(dfall)
# 	n_col = len(dfall[0].columns)
# 	n_ind = len(dfall[0].index)
# 	fig, axe = plt.subplots(1, figsize=myfigsize)
#
# 	mycolor_palette = sns.color_palette("tab10", n_col)
# 	cmap=ListedColormap(mycolor_palette.as_hex())
#
# 	for df in dfall : # for each data frame
# 		axe = df.plot(kind="bar",
# 					  linewidth=0,
# 					  stacked=True,
# 					  ax=axe,
# 					  legend=False,
# 					  grid=False,
# 					  colormap=cmap,
# 					  fontsize=fontsize,
# 					  **kwargs)  # make bar plots
#
# 	h,l = axe.get_legend_handles_labels() # get the handles we want to modify
# 	for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
# 		for j, pa in enumerate(h[i:i+n_col]):
# 			for rect in pa.patches: # for each index
# 				rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
# 				rect.set_hatch(H * int(i / n_col)) #edited part
# 				if n_df != 1:
# 					rect.set_width(1 / float(n_df + 1))
#
# 	# axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
# 	if xticklabels is None:
# 		axe.set_xticklabels(df.index, rotation=90)
# 	elif xticklabels== False:
# 		axe.set_xticklabels([], rotation=90)
# 	else:
# 		if not xticksxpos is None:
# 			axe.set_xticks(xticksxpos)
# 			axe.set_xticklabels(xticklabels, rotation=90)
# 		else:
# 			axe.set_xticklabels(xticklabels, rotation=90)
# 	axe.set_title(title)
#
# 	# Add invisible data to add another legend
# 	n=[]
# 	for i in range(n_df):
# 		n.append(axe.bar(0, 0, color="gray", hatch=H * i))
#
# 	l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
# 	if labels is not None:
# 		l2 = plt.legend(n, labels, loc=[1.01, 0.1])
# 	else:
# 		l2 = plt.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
#
# 	# axe.add_artist(l1)
#
# 	if savePlot:
# 		fig.savefig(fpath, bbox_extra_artists=(l2,), bbox_inches='tight')


def get_chr_ticks(genes_positions_table, data, id_col='id', chr_col='chr'):
    # make "id" the index for faster lookup
    genes_positions_table = genes_positions_table.set_index([id_col]).copy()
    # get only the labels that exist in the data
    labels = [genes_positions_table.loc[x][chr_col] for x in data.columns]

    from natsort import natsorted
    # get the unique labels and order them for the xticks
    xticks = np.array(natsorted(set(labels)))

    # count how many genes in the data for each label
    chr_size = pd.Series(labels).value_counts()
    # reorder the labels counts as xticks
    chr_size = chr_size.loc[natsorted(chr_size.index)]

    # the cumulative sum to get the position of the column when each label ends
    chr_endpos = chr_size.cumsum()

    return xticks, chr_endpos


def break_yAxis(bottom_axis, top_axis, d=0.005):
    # leave as is
    bottom_axis.spines['bottom'].set_visible(False)
    top_axis.spines['top'].set_visible(False)
    bottom_axis.xaxis.tick_top()
    bottom_axis.tick_params(labeltop='off')
    top_axis.xaxis.tick_bottom()

    kwargs = dict(transform=bottom_axis.transAxes, color='k', clip_on=False)
    bottom_axis.plot((-d, +d), (-d, +d), **kwargs)
    bottom_axis.plot((1-d, 1+d), (-d, +d), **kwargs)
    kwargs.update(transform=top_axis.transAxes)
    top_axis.plot((-d, +d), (1-d, 1+d), **kwargs)
    top_axis.plot((1-d, 1+d), (1-d, 1+d), **kwargs)


def distplot_breakYaxis(
        x, ymax_bottom, ymax_top, mytitle='', color=None, d=0.005, pad=0,
        figsize=(10, 5)):

    f, axis = plt.subplots(2, 1, sharex=True, figsize=figsize)
    # choose your plot
    sns.distplot(x.flatten(), ax=axis[0], hist=True, kde=False, color=color)
    sns.distplot(x.flatten(), ax=axis[1], hist=True, kde=False, color=color)

    # set limitis on y axis (play around with threshold)
    axis[0].set_ylim(ymax_top-ymax_bottom, ymax_top)
    axis[0].set_title(mytitle)
    axis[1].set_ylim(0, ymax_bottom)

    # leave as is
    break_yAxis(axis[0], axis[1], d=d)
    plt.tight_layout(pad=pad)


def plot_heatmap_custom(
        data, figsize=(20, 3), vmin=None, vmax=None, xticklabels=False,
        yticklabels=False, xlabel='', ylabel='', title='', cmap=None,
        xticks_xlabels=None, xticks_xpos=None, square=False):

    plt.figure(figsize=figsize)

    sns.heatmap(
        data, vmin=vmin, vmax=vmax, xticklabels=xticklabels,
        yticklabels=yticklabels, cmap=cmap, square=square,
        annot_kws={"size": 50})
    if not(xticks_xlabels is None):
        plt.xticks(xticks_xpos, xticks_xlabels, rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return plt


def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='red', midcol='white', maxcol='blue'):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    cmap = LinearSegmentedColormap.from_list(
        name=name, colors=[mincol, midcol, maxcol], N=numcolors)
    return cmap


def plot_hist_and_heatmap(
        ar, arMax=None, arMin=None, arStep=None, xlabel='', ylabel='',
        xticklabels=False, yticklabels=False, xlabels=None, xpos=None,
        saveimg=False, showimg=False, imgpath='./'):

    if saveimg and not showimg:
        plt.ioff()

    # histogram
    if arMax is None:
        arMax = ar.max()
    if arMin is None:
        arMin = ar.min()
    # arStep = 1
    # arStep = ar.std()
    if arMin != arMax:
        if arStep is None:
            arStep = (arMax-arMin)*0.2
        # arStep = np.round(max((arMax-arMin)*0.1, ar.std(), 1))
        # if arMax <= 1:
        #   arStep = min(max((arMax-arMin)*0.1, ar.std()), 0.1) #arStep = 0.1
        bin_width = 0.5
        bins = np.arange(
            arMin-(arStep*bin_width), arMax+(arStep/bin_width),
            arStep*bin_width)

        cmap_step = min(arStep, 1)
        cmap_node = custom_div_cmap(
            int((arMax-arMin+cmap_step)/cmap_step),
            mincol='yellow', midcol='orange', maxcol='darkred')

        # distribution
        fig = plt.figure(figsize=(10, 3))
        sns.distplot(ar, hist=False, kde=True)
        plt.axvline(x=ar.mean(), color='r', linestyle='--')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('distribution of '+ylabel)
        fig.tight_layout()
        if saveimg:
            plt.savefig(imgpath+'a1.png')
        if showimg:
            plt.show()
        plt.close('all')

        # histogram
        fig = plt.figure(figsize=(10, 3))
        sns.distplot(
            ar, hist=True, kde=False,
            hist_kws={'align': 'left', 'rwidth': bin_width},
            bins=bins)
        plt.axvline(x=ar.mean(), color='r', linestyle='--')
        plt.xticks(np.arange(arMin, arMax+arStep, arStep))
        plt.xlim(arMin-arStep, arMax+arStep)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('histogram of '+ylabel)
        fig.tight_layout()
        if saveimg:
            plt.savefig(imgpath+'a2.png')
        if showimg:
            plt.show()
        plt.close('all')

        # heatmap
        x = ar.values.reshape(1, len(ar))

        if xlabels is not None:
            xticklabels, yticklabels = False, False
        fig = plot_heatmap_custom(
            x, (20, 2), arMin, arMax, xticklabels, yticklabels,
            '', '', '', cmap_node, xticks_xlabels=xlabels, xticks_xpos=xpos)
        fig.tight_layout()
        if saveimg:
            fig.savefig(imgpath+'b.png')
        if showimg:
            plt.show()
        plt.close('all')
    else:
        logger.warning(
            ' -- !!! WARNING !!! -- ' +
            'the values do not vary for: '+xlabel +
            '-----------------------')


def plot_dist_and_heatmap(
        data, genes_positions_table, title='', isMut=False, cmap='div',
        saveimg=False, showimg=False, imgpath='./'):

    n_rows = data.shape[0]

    if saveimg and not showimg:
        plt.ioff()

    # show cnv values distribution and heatmaps
    fig = plt.figure(figsize=(12, 4))
    sns.distplot(data.values.flatten(), hist=False)
    plt.title(title+' values distribution')
    fig.tight_layout()
    if saveimg:
        plt.savefig(imgpath+'a.png')
    if showimg:
        plt.show()
    plt.close('all')

    if genes_positions_table is None:
        xlabels, xpos = None, None
        xticklabels = int(data.shape[1] / 10)
    else:
        xlabels, xpos = get_chr_ticks(genes_positions_table, data)
        xticklabels = False

    RdBu_custom = custom_div_cmap(
        1000, mincol='blue', midcol='white', maxcol='red')
    YlOrRd_custom = custom_div_cmap(
        1000, mincol='white', midcol='orange', maxcol='red')
    if cmap == 'div':
        cmap = RdBu_custom  # "RdBu_r"
    elif cmap == 'pos':
        cmap = "YlOrRd"
    else:
        cmap = cmap

    if isMut:
        vmax = [None, 2, 1]
        vmin = [0, 0, 0]
        cmap = YlOrRd_custom
    else:
        vmax = [
            None,
            data.values.flatten().mean() + 2*data.values.flatten().std(),
            data.values.flatten().mean() + data.values.flatten().std()]
        vmin = [None, -vmax[1], -vmax[2]]

    fig = plot_heatmap_custom(
        data, figsize=(20, 3),
        vmin=vmin[0], vmax=vmax[0], xticklabels=xticklabels, yticklabels=False,
        xlabel='', ylabel='', title=title+'heatmap',
        cmap=cmap, xticks_xlabels=xlabels, xticks_xpos=xpos,
        square=False)
    fig.tight_layout()
    if saveimg:
        fig.savefig(imgpath+'b.png')
    if showimg:
        plt.show()
    plt.close('all')
    #
    # plot_heatmap_custom(
    #    data, figsize=(20,3),
    #     vmin=vmin[1], vmax=vmax[1], xticklabels=False, yticklabels=False,
    #     xlabel='', ylabel='', title='',
    #     cmap=cmap, xticks_xlabels=xlabels, xticks_xpos=xpos,
    #     square=False)
    #
    # plot_heatmap_custom(
    #    data, figsize=(20,3),
    #     vmin=vmin[2], vmax=vmax[2], xticklabels=False, yticklabels=False,
    #     xlabel='', ylabel='', title='',
    #     cmap=cmap, xticks_xlabels=xlabels, xticks_xpos=xpos,
    #     square=False)


def plot_DL_diagnostics(
      data, recon_data, params, rg1=0, heatmap_thres=None, cmap_custom=None,
      genes_positions_table=None, h_figsize=(20, 5), p_figsize=(12, 4),
      saveimg=False, imgpath='./'):
    n_rows = data.shape[0]

    # with the rg1 we choose to show the std(rg1=0)
    # or the sem(rg1=1) together with the mean
    rg_text = ['std', 'sem']
    # sparsity threshold
    s0 = params['s0']
    # error threshold
    e0 = params['e0']
    # max D size as a float (percentage)
    maxD0 = params['maxD0']
    # threshold for regression difference
    # for stopping criterion in dictionary optimization
    reg0 = params['reg0']
    # percentage of data cols to be optimized at each iteration
    w0 = params['w0']
    # number of new atoms to be added to the Dictionary at each iteration
    # (actually a0-1 atoms are added
    # because at each oteration one atom is removed)
    a0 = params['a0']
    # number of iterations after which a full ksvd will be called
    # (vs. the stagewise that is called at every other iteration)
    full0 = params['full0']
    # number of iterations the full ksvd will run for
    full1 = params['full1']

    # get RE from results
    RE_iter = params['RE_iter']
    RE_window = params['RE_window']
    num = params['log_iter'].shape[0]

    # define colors for Frobenius norm and mean error
    from matplotlib import colors
    m_color = 'forestgreen'
    m_area_color = 'azure'

    # print the window means ?
    toPrint = True

    # plot heatmaps
    if heatmap_thres is None:
        heatmap_thres = abs(data).max().max()

    if data.min().min() < 0:
        thres = [-heatmap_thres, heatmap_thres]
        if cmap_custom is None:
            cmap_custom = custom_div_cmap(
                1000, mincol='blue', midcol='white', maxcol='red')
    else:
        thres = [0, heatmap_thres]
        if cmap_custom is None:
            cmap_custom = custom_div_cmap(
                1000, mincol='yellow', midcol='orange', maxcol='red')
    if genes_positions_table is not None:
        xticks_xlabels, xticks_xpos = get_chr_ticks(
            genes_positions_table, data)
    else:
        xticks_xlabels, xticks_xpos = None, None
    # original data

    h = plot_heatmap_custom(
        data, figsize=h_figsize,
        vmin=thres[0], vmax=thres[1], xticklabels=False, yticklabels=False,
        xlabel='', ylabel='', title='original data',
        cmap=cmap_custom, xticks_xlabels=xticks_xlabels,
        xticks_xpos=xticks_xpos, square=False)
    if saveimg:
        h.savefig(
            imgpath+'1.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)

    # reconstructed data
    title = 'reconstr. [s0, e0, maxD0, reg0, w0, a0, full0, full1] = ' +\
        str([s0, e0, maxD0, reg0, w0, a0, full0, full1])
    # recon_data = pd.DataFrame(
    #   np.dot(D,X), columns=data.columns, index=data.index)
    h = plot_heatmap_custom(
        recon_data, figsize=h_figsize,
        vmin=thres[0], vmax=thres[1], xticklabels=False, yticklabels=False,
        xlabel='', ylabel='', title=title,
        cmap=cmap_custom, xticks_xlabels=xticks_xlabels,
        xticks_xpos=xticks_xpos, square=False)
    if saveimg:
        h.savefig(
            imgpath+'2.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)

    # compare RE and RE worst using the same axis
    sameaxis = True

    # RE_iter: [mean, std, sem, worst_mean, worst_std, worst_sem,
    #           #atoms, abs max from residuals]
    # RE_window: [mean of means, mean of stds, mean of sems,
    #             mean of worst_means, mean of worst_stds, mean of worst_sems,
    #             window_size, #atoms (when window was calculated)]
    # compute MAX btwn RE and RE worst to compare in common axis
    RErg0_commonMax = RE_iter[:, [0, 3]].max()
    # std/sem and worst_std/worst_sem (for the mean+std plot)
    RErg1_commonMax = RE_iter[:, [1, 4+rg1]].max()

    REWrg0_commonMax = RE_window[:, [0, 3]].max()
    # std/sem and worst_std/worst_sem (for the mean+std plot)
    REWrg1_commonMax = RE_window[:, [1, 4+rg1]].max()
    if sameAxis:
        # DIFF mean and worst_mean
        REmdiff_commonMax = max([
            abs(np.diff(RE_iter[:, 0])).max(),
            abs(np.diff(RE_iter[:, 3])).max()])
        # DIFF std and worst_std
        REstddiff_commonMax = max([
            abs(np.diff(RE_iter[:, 1])).max(),
            abs(np.diff(RE_iter[:, 4])).max()])
        # DIFF sem and worst_sem
        REsemdiff_commonMax = max([
            abs(np.diff(RE_iter[:, 2])).max(),
            abs(np.diff(RE_iter[:, 5])).max()])

        # window DIFF mean and worst_mean
        REWmdiff_commonMax = max([
            abs(np.diff(RE_window[:, 0])).max(),
            abs(np.diff(RE_window[:, 3])).max()])
        # window DIFF std and worst_std
        REWstddiff_commonMax = max([
            abs(np.diff(RE_window[:, 1])).max(),
            abs(np.diff(RE_window[:, 4])).max()])
        # window DIFF sem and worst_sem
        REWsemdiff_commonMax = max([
            abs(np.diff(RE_window[:, 2])).max(),
            abs(np.diff(RE_window[:, 5])).max()])

    # subplots
    f3, axarr3 = plt.subplots(
        3, 2, sharex=True, figsize=(p_figsize[0], p_figsize[0]))
    f4, axarr4 = plt.subplots(
        3, 2, sharex=True, figsize=(p_figsize[0], p_figsize[0]))

    # repeat for RE and RE_worst
    # for the RE_iter indexing,
    # because the RE and RE_worst corresponding values
    # are 3 places away from each other
    for er in [0, 3]:
        n = 1  # param for diff
        if er == 0:
            ertype = 'RE'
            col = 0
        else:  # er == 3
            ertype = 'RE worst'
            col = 1

        # plot RE mean with + std
        # plor mean
        axarr3[0, col].plot(RE_iter[:, er+0], color=m_color)
        # plot +std/sem
        plus_intrvl = RE_iter[:, er+0] + RE_iter[:, er+1+rg1]
        axarr3[0, col].plot(
            plus_intrvl, linestyle='--', linewidth=1, color=m_color)
        # plot -std/sem (but do not show if negative values)
        minus_intrvl = RE_iter[:, er+0] - RE_iter[:, er+1+rg1]
        minus_intrvl[minus_intrvl < 0] = 0
        axarr3[0, col].plot(
            minus_intrvl, linestyle='--', linewidth=1, color=m_color)
        # fill the interval with color
        axarr3[0, col].fill_between(
            np.arange(len(RE_iter[:, er+0])), plus_intrvl, minus_intrvl,
            facecolor=m_area_color, edgecolor='')
        # axes and title
        # add line to show zero
        axarr3[0, col].axhline(y=0, color='k', linestyle='--')
        # set ylim (use the value for the common y axis)
        axarr3[0, col].axis([
            0, RE_iter.shape[0], -0.1, RErg0_commonMax+RErg1_commonMax])
        axarr3[0, col].set_title(ertype+' mean and '+rg_text[rg1])  # title

        for n in [1, 2]:
            # plot DIFF of RE mean and std or sem
            # diff of mean
            axarr3[n, col].plot(
                np.arange(len(RE_iter[:, er+0])-n),
                np.diff(RE_iter[:, er+0], n=n),
                color=m_color, linewidth=1.5)
            # diff of std/sem
            axarr3[n, col].plot(
                np.arange(len(RE_iter[:, er+1+rg1])-n),
                np.diff(RE_iter[:, er+1+rg1], n=n),
                color=m_color, linestyle='--', linewidth=2)
            theMax = max(REstddiff_commonMax, REmdiff_commonMax)
            if rg1 == 1:
                theMax = max(REsemdiff_commonMax, REmdiff_commonMax)
            if sameAxis:
                axarr3[n, col].axis([0, RE_iter.shape[0]-n, -theMax, theMax])
            axarr3[n, col].axhline(y=0, color='k', linestyle='--')
            axarr3[n, col].set_title(
                ertype+' diff(n='+str(n)+') mean and '+rg_text[rg1])

        # plot RE mean: windows mean+-std
        # plor mean
        axarr4[0, col].plot(
            np.arange(RE_window.shape[0]), (RE_window[:, er+0]), color=m_color)
        # plot +std/sem
        plus_intrvl = (RE_window[:, er+0]) + (RE_window[:, er+1+rg1])
        axarr4[0, col].plot(
            np.arange(RE_window.shape[0]), plus_intrvl,
            linestyle='--', linewidth=1, color=m_color)
        # plot -std/sem (but do not show if negative values)
        minus_intrvl = (RE_window[:, er+0]) - (RE_window[:, er+1+rg1])
        minus_intrvl[minus_intrvl < 0] = 0
        axarr4[0, col].plot(
            np.arange(RE_window.shape[0]), minus_intrvl,
            linestyle='--', linewidth=1, color=m_color)
        # fill the interval with color
        axarr4[0, col].fill_between(
            np.arange(RE_window.shape[0]), plus_intrvl, minus_intrvl,
            facecolor=m_area_color, edgecolor='')
        # plot scatter of the mean
        # (less point here than in RE_iter so let's see them)
        axarr4[0, col].scatter(
            np.arange(RE_window.shape[0]), (RE_window[:, er+0]), color=m_color)
        # axes and title
        axarr4[0, col].axhline(y=0, color='k', linestyle='--')
        # set ylim (use the value for the common y axis)
        axarr4[0, col].axis(
            [0, RE_window.shape[0], -0.1, REWrg0_commonMax+REWrg1_commonMax])
        axarr4[0, col].set_title('window '+ertype+' mean and '+rg_text[rg1])

        for n in [1, 2]:
            # plot DIFF of RE window mean and std or sem
            # diff of mean
            axarr4[n, col].plot(
                np.arange(len(RE_window[:, er+0])-n),
                np.diff(RE_window[:, er+0], n=n), color=m_color)
            # diff of std/sem
            axarr4[n, col].plot(
                np.arange(len(RE_window[:, er+1+rg1])-n),
                np.diff(RE_window[:, er+1+rg1], n=n),
                color=m_color, linestyle='--', linewidth=2)
            theMax = max(REWstddiff_commonMax, REWmdiff_commonMax)
            if rg1 == 1:
                theMax = max(REWsemdiff_commonMax, REWmdiff_commonMax)
            if sameAxis:
                axarr4[n, col].axis([0, RE_window.shape[0]-n, -theMax, theMax])
            axarr4[n, col].axhline(y=0, color='k', linestyle='--')
            axarr4[n, col].set_title(
                'window '+ertype+' diff(n='+str(n)+') mean and '+rg_text[rg1])

    if saveimg:
        f3.savefig(
            imgpath+'3.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
    if saveimg:
        f4.savefig(
            imgpath+'4.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)

    x = data.values
    plt.figure(figsize=p_figsize)
    sns.distplot(x.flatten(), hist=False, color='k')
    y = recon_data.values  # np.dot(D,X)
    sns.distplot(
        y.flatten(), hist=False, color='r', kde_kws={"linestyle": '--'})
    plt.title('distribution of original(black) and reconstructed(red) data')
    plt.tight_layout()
    if saveimg:
        plt.savefig(
            imgpath+'5.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)

    plt.figure(figsize=p_figsize)
    z = (x - y)
    sns.distplot(z.flatten(), hist=False, color='b')
    plt.title('distribution of difference (original - reconstructed)')
    plt.tight_layout()
    if saveimg:
        plt.savefig(
            imgpath+'6.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)

    plt.figure(figsize=p_figsize)
    z = abs(x - y)
    sns.distplot(z.flatten(), hist=False, color='b')
    plt.title('distribution of absolute difference')
    plt.tight_layout()
    if saveimg:
        plt.savefig(
            imgpath+'7.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)

    return


def plot_DL_reconst_quality(
        data, D, data_rec, geneCoord=None, plotOthers=True, corr="P",
        figsize=(10, 10), saveimg=False, imgpath='./'):
    imgCount = 0

    prwD = pdist(D)
    prwDATA = pdist(data.values)
    prwDATA_rec = pdist(data_rec)

    if plotOthers:
        plt.figure(figsize=(figsize[0], figsize[0]-2))
        sns.heatmap(
            squareform(prwDATA), xticklabels=False,
            yticklabels=False, square=True)
        plt.title('pairwise Euclidean distances between patients in DATA')
        if saveimg:
            imgCount += 1
            plt.savefig(
                imgpath+str(imgCount)+'.png', transparent=True,
                bbox_inches='tight', pad_inches=0.1, frameon=False)

        plt.figure(figsize=(figsize[0], figsize[0]-2))
        sns.heatmap(
            squareform(prwDATA), xticklabels=False,
            yticklabels=False, square=True)
        plt.title(
            'pairwise Euclidean distances between patients ' +
            'in reconstructed DATA')
        if saveimg:
            imgCount += 1
            plt.savefig(imgpath+str(
                imgCount)+'.png', transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)

        plt.figure(figsize=(figsize[0], figsize[0]-2))
        sns.heatmap(
            squareform(prwD), xticklabels=False,
            yticklabels=False, square=True)
        plt.title(
            'pairwise Euclidean distances between patients in Dictionary')
        if saveimg:
            imgCount += 1
            plt.savefig(
                imgpath+str(imgCount)+'.png', transparent=True,
                bbox_inches='tight', pad_inches=0.1, frameon=False)

    if corr == "P":
        corr_name_ = 'Pearson'
        samplescorr1_ = sp.stats.pearsonr(prwD, prwDATA)[0]
        samplescorr2_ = sp.stats.pearsonr(prwDATA, prwDATA_rec)[0]

        corr_sets = [(corr_name_, samplescorr1_, samplescorr2_)]
    elif corr == "S":
        corr_name_ = 'Spearman'
        samplescorr1_ = sp.stats.spearmanr(prwD, prwDATA)[0]
        samplescorr2_ = sp.stats.spearmanr(prwDATA, prwDATA_rec)[0]

        corr_sets = [(corr_name_, samplescorr1_, samplescorr2_)]
    else:
        corr_name_a = 'Pearson'
        samplescorr1_a = sp.stats.pearsonr(prwD, prwDATA)[0]
        samplescorr2_a = sp.stats.pearsonr(prwDATA, prwDATA_rec)[0]

        corr_name_b = 'Spearman'
        samplescorr1_b = sp.stats.spearmanr(prwD, prwDATA)[0]
        samplescorr2_b = sp.stats.spearmanr(prwDATA, prwDATA_rec)[0]

        corr_sets = [
            (corr_name_a, samplescorr1_a, samplescorr2_a),
            (corr_name_b, samplescorr1_b, samplescorr2_b)]

    for corr_name, samplescorr1, samplescorr2 in corr_sets:
        plt.figure(figsize=figsize)
        plt.scatter(prwD, prwDATA)
        plt.xlabel('pairw. eucl. dist btwn DICTIONARY patients')
        plt.ylabel('pairw. eucl. dist btwn DATA patients')
        # plt.text(
        #   0,prwDATA.max(),
        #   'Pearson Correlation Coef. = '+str(round(samplescorr1,4))
        plt.annotate(
            corr_name+' Correlation Coef. = '+str(round(samplescorr1, 4)),
            (0, 0), (0, -50), xycoords='axes fraction',
            textcoords='offset points', va='top', fontsize=18)
        if saveimg:
            imgCount += 1
            plt.savefig(
                imgpath+str(imgCount)+corr_name+'.png', transparent=True,
                bbox_inches='tight', pad_inches=0.1, frameon=False)

        plt.figure(figsize=figsize)
        plt.scatter(prwDATA, prwDATA_rec)
        plt.xlabel('pairw. eucl. dist btwn DATA patients')
        plt.ylabel('pairw. eucl. dist btwn DATA reconstr. patients')
        # plt.text(
        #   0,prwDATA_rec.max(),
        #   'Pearson Correlation Coef. = '+str(round(samplescorr2,4)))
        plt.annotate(
            corr_name+' Correlation Coef. = '+str(round(samplescorr2, 4)),
            (0, 0), (0, -50), xycoords='axes fraction',
            textcoords='offset points', va='top', fontsize=18)
        if saveimg:
            imgCount += 1
            plt.savefig(
                imgpath+str(imgCount)+corr_name+'.png', transparent=True,
                bbox_inches='tight', pad_inches=0.1, frameon=False)

    if plotOthers:
        # RE per gene/column
        R = data - data_rec
        RE_cols_m = abs(R).mean(axis=0)
        RE_cols_std = abs(R).std(axis=0).to_frame()
        RE_cols_std.insert(0, 'minus', RE_cols_std.iloc[:, 0])
        RE_cols_std['minus'][RE_cols_m-RE_cols_std['minus'] < 0] = \
            RE_cols_m[RE_cols_m-RE_cols_std['minus'] < 0]
        RE_cols_std = RE_cols_std.T

        plt.figure(figsize=(figsize[0]*2.5, figsize[0]*0.5))
        plt.errorbar(
            np.arange(RE_cols_m.shape[0]), RE_cols_m.values,
            xerr=0, yerr=RE_cols_std.values, fmt='o',
            ms=1.5, mfc='k', ecolor='gray', elinewidth=0.05)
        plt.axhline(y=RE_cols_m.mean(), color='r', linestyle='-', linewidth=3)
        plt.xlim(0, RE_cols_m.shape[0])
        plt.title('RE per gene/column')
        if geneCoord is not None:
            xlabels, xpos = geneCoord
            plt.xticks(xpos, xlabels, rotation=0)
        if saveimg:
            imgCount += 1
            plt.savefig(
                imgpath+str(imgCount)+'.png', transparent=True,
                bbox_inches='tight', pad_inches=0.1, frameon=False)

    return corr_sets


def plot_interDX(
        data, genes_positions_table, dxl, newDir, clean_params=None,
        saveimg=False, imgdir='./', figsize=(25, 5), heatmap_thres=None):
    from prepare_data.process_data import reverse_preprocessing

    if heatmap_thres is None:
        heatmap_thres = abs(data).max().max()
    if data.min().min() < 0:
        thres = [-heatmap_thres, heatmap_thres]
        cmap_custom = custom_div_cmap(
            1000, mincol='blue', midcol='white', maxcol='red')
    else:
        thres = [0, heatmap_thres]
        cmap_custom = custom_div_cmap(
            1000, mincol='yellow', midcol='orange', maxcol='red')

    xticks_xlabels, xticks_xpos = get_chr_ticks(genes_positions_table, data)
    for i, l in enumerate(dxl):
        logger.info(str(l[0]))
        D = l[1]
        X = l[2]

        title = 'reconstr. at '+l[0]
        recon_data = pd.DataFrame(
            np.dot(D, X), columns=data.columns, index=data.index)
        if clean_params is not None:
            raw_mean, raw_std, data_type = clean_params
            recon_data = reverse_preprocessing(
                recon_data, raw_mean, raw_std, data_type=data_type)
        h = plot_heatmap_custom(
            recon_data, figsize=figsize,
            vmin=None, vmax=None, xticklabels=False, yticklabels=False,
            xlabel='', ylabel='', title=title,
            cmap=cmap_custom, xticks_xlabels=xticks_xlabels,
            xticks_xpos=xticks_xpos, square=False)
        if saveimg:
            h.savefig(
                newDir+'inter_recon'+str(i)+'__'+l[0]+'.png', transparent=True,
                bbox_inches='tight', pad_inches=0.1, frameon=False)
    return


def fgrid_set_labels_rotation(g, degrees, which='y'):
    if which is 'y':
        for ax in g.axes.flat:
            for label in ax.get_yticklabels():
                label.set_rotation(degrees)
    elif which is 'x':
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(degrees)
    else:
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(degrees[0])
            for label in ax.get_yticklabels():
                label.set_rotation(degrees[1])


# def plot_aggr_cnv(aggr_pos,aggr_neg,
# 				  xlabels, xpos,
# 				  xlabel='', title='',
# 				  myfigsize=(20,3)):
# 	s = aggr_pos*100
# 	len_s = len(s)
# 	plt.figure(figsize=myfigsize)
# 	plt.xlim(-1, len_s)
# 	markerline1, stemlines1, baseline1 =plt.stem(
#       np.arange(0,len_s), s, basefmt='k-')
# 	plt.setp(markerline1, markerfacecolor= 'red', markersize =0)
# 	plt.setp(stemlines1, linewidth=0.5, color=plt.getp(
#       markerline1,'markerfacecolor'))
#
# 	s = aggr_neg*100
# 	markerline2, stemlines2, _ =plt.stem(np.arange(0,len_s), -s, basefmt='k-')
# 	plt.setp(markerline2, markerfacecolor= 'blue', markersize =0)
# 	plt.setp(stemlines2, linewidth=0.5, color=plt.getp(
#       markerline2,'markerfacecolor'))
#
# 	plt.xticks(xpos,xlabels, rotation=0)
# 	plt.xlabel(xlabel)
# 	plt.ylabel('%')
# 	plt.ylim([-100,100])
# 	plt.title(title)


def plot_aggr_freq(
        aggr_ampl, aggr_del,
        xlabels, xpos,
        xlabel=(
            'chromosomes ' +
            '(the number is aligned at the end of the chr region)'),
        title='', printNames=False,
        font=2, height_space=1,
        del_space=50, ampl_space=50,
        figsize=(20, 5)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.axhline(y=0, c='k', linewidth=0.5)
    maxLenGeneName = max(
        len(max(aggr_ampl.index.values, key=len)),
        len(max(aggr_del.index.values, key=len)))

    ampl_space = aggr_ampl.shape[0] * 0.0025
    del_space = aggr_del.shape[0] * 0.0025
    ##
    s = aggr_ampl*100
    sMax = s.max()
    y_offset = (sMax*height_space)+maxLenGeneName
    if sMax+y_offset > 100:
        y_offset = 100-sMax
    xs = s.nonzero()[0]
    n = len(xs)
    if n > 0:
        # step = xs.std()*ampl_space
        step = ampl_space
        mid_x = int(-1+n/2)
        new_xs = np.ndarray(n)
        count = 1
        for i in np.arange(mid_x-1, -1, -1):
            new_xs[i] = xs[i]-count*step
            count = count + 1
        new_xs[mid_x] = xs[mid_x]
        count = 1
        for i in np.arange(mid_x+1, n):
            new_xs[i] = xs[i]+count*step
            count = count + 1

    len_s = len(s)
    ax.set_xlim(0, len_s)
    ar = np.arange(0, len_s)
    plt.xlim(0, len_s)
    # plt.stem(
    #   np.arange(0,len_s), s, linefmt='b-', markerfmt='b ', basefmt='k-')
    plt.bar(x=np.arange(0, len_s), height=s, width=1, color='b')
    if printNames:
        for i, x in enumerate(xs):
            geneName = aggr_ampl.iloc[x:x+1].index.values[0]
            ax.annotate(
                '%s' % geneName, xy=(new_xs[i], s[x]+y_offset),
                textcoords='data', fontsize=font, rotation=90,
                horizontalalignment='center', verticalalignment='top')

    ##
    s = aggr_del*100
    sMax = s.max()
    y_offset = (sMax*height_space)+maxLenGeneName
    if sMax+y_offset > 100:
        y_offset = 100-sMax
    xs = s.nonzero()[0]
    n = len(xs)
    if n > 0:
        # step = xs.std()*del_space
        step = del_space
        mid_x = int(-1+n/2)
        new_xs = np.ndarray(n)
        count = 1
        for i in np.arange(mid_x-1, -1, -1):
            new_xs[i] = xs[i]-count*step
            count = count + 1
        new_xs[mid_x] = xs[mid_x]
        count = 1
        for i in np.arange(mid_x+1, n):
            new_xs[i] = xs[i]+count*step
            count = count + 1
    ##
    # plt.stem(
    #   np.arange(0,len_s), -s, linefmt='r-', markerfmt='r ', basefmt='k-')
    plt.bar(x=np.arange(0, len_s), height=-s, width=1, color='r')
    if printNames:
        for i, x in enumerate(xs):
            geneName = aggr_del.iloc[x:x+1].index.values[0]
            ax.annotate(
                '%s' % geneName, xy=(new_xs[i], -s[x]-y_offset),
                textcoords='data', fontsize=font, rotation=90,
                horizontalalignment='center', verticalalignment='bottom')

    if xpos is not None:
        plt.xticks(xpos, xlabels, rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel('%')
    plt.ylim([-100, 100])
    plt.title(title)


def atoms_data(
        av_fromD, idx_patients, av_fromX, idx_genes,
        which, data, D, X,
        xlabels, xpos,
        RdBu_custom, PuOr_custom,
        mySize=(12, 3)):

    plot_heatmap_custom(D[:, which:which+1], (1, mySize[1]),
                        -1, 1, False, False,
                        str(which),
                        '',
                        'atom from D',
                        RdBu_custom)

    plot_heatmap_custom(X[which:which+1, :], (mySize[0], 1),
                        -4, 4, False, False,
                        '',
                        str(which),
                        'atom from X',
                        PuOr_custom)

    data = data.copy()
    R = data - np.dot(D, X)
    Derr = abs(R).mean(axis=0)
    patientCount = data.shape[0]
    geneCount = data.shape[1]

    atom_toPlot = which

    if type(atom_toPlot) is np.ndarray:
        if len(atom_toPlot) > 1:
            logger.warning('more than one atoms, plotting the first...')
            atom_toPlot = atom_toPlot[0]
        elif len(atom_toPlot) == 0:
            logger.error('no atom found!')
            return

    logger.info(
        'Atom: '+str(atom_toPlot) +
        ' -with value: '+str(av_fromD[atom_toPlot])+' from D' +
        ' -with value: '+str(av_fromX[atom_toPlot])+' from X')

    idx_genes = idx_genes[atom_toPlot]
    logger.info('Number of genes with this atom: '+str(len(idx_genes)))
    idx_patients = idx_patients[atom_toPlot]
    if type(idx_patients) is np.ndarray:
        lenPatients = len(idx_patients)
        if lenPatients == 1:
            idx_patients = idx_patients[0]
        logger.info('Number of patients with this atom: '+str(lenPatients))
    else:
        lenPatients = 1
        logger.info(
            'Most important patient with this atom: '+str(idx_patients))

    set_allGenes = set(range(geneCount))
    set_myGenes = set(idx_genes)
    set_diffGenes = set.difference(set_allGenes, set_myGenes)
    len(set_diffGenes)

    data.iloc[:, list(set_diffGenes)] = 0

    # plot the heatmap ONLY my genes
    plot_heatmap_custom(data, mySize,
                        -2, 2, False, False,
                        'chromosomes',
                        'samples',
                        'selected genes',
                        RdBu_custom,
                        xlabels, xpos)

    # plot the heatmap ONLY my genes AND my patients
    if lenPatients == 1:
        plot_heatmap_custom(
            data.iloc[idx_patients:idx_patients+1, :],
            (mySize[0], min(3, lenPatients)),
            -2, 2, False, False,
            'chromosomes',
            str(idx_patients),
            'selected genes AND patients',
            RdBu_custom,
            xlabels, xpos)
    else:
        plot_heatmap_custom(
            data.iloc[idx_patients, :], (mySize[0], min(3, lenPatients)),
            -2, 2, False, False,
            'chromosomes',
            str(idx_patients),
            'selected genes AND patients',
            RdBu_custom,
            xlabels, xpos)

    # plot the RE ONLY mygenes
    plt.figure(figsize=mySize)
    plt.scatter(
        list(set_diffGenes), Derr[list(set_diffGenes)], c="k", edgecolor='')
    plt.scatter(
        list(set_myGenes), Derr[list(set_myGenes)], c="orange", edgecolor='')
    plt.axis([0, geneCount, 0, Derr.max()])
    plt.xticks(xpos, xlabels, rotation=0)
    plt.xlabel('chromosomes')
    plt.ylabel('samples')


def plot_errorbars(m, inCols=True, toSort=False, newFig=True, **plot_kwargs):

    plot_kwargs['figsize'] = plot_kwargs.get('figsize', (25, 8))
    plot_kwargs['fmt'] = plot_kwargs.get('fmt', 's')
    plot_kwargs['ms'] = plot_kwargs.get('ms', 6)
    plot_kwargs['mfc'] = plot_kwargs.get('mfc', 'k')
    plot_kwargs['ecolor'] = plot_kwargs.get('ecolor', 'gray')
    plot_kwargs['elinewidth'] = plot_kwargs.get('elinewidth', 3)
    plot_kwargs['draw_axhline'] = plot_kwargs.get('draw_axhline', False)
    plot_kwargs['set_axhline'] = plot_kwargs.get('set_axhline', None)
    plot_kwargs['axhline_linestyle'] = plot_kwargs.get(
        'axhline_linestyle', '--')
    plot_kwargs['axhline_linewidth'] = plot_kwargs.get('axhline_linewidth', 2)
    plot_kwargs['axhline_color'] = plot_kwargs.get('axhline_color', 'r')
    plot_kwargs['ylimits'] = plot_kwargs.get('ylimits', None)
    plot_kwargs['xticks'] = plot_kwargs.get('xticks', None)
    plot_kwargs['highlight'] = plot_kwargs.get('highlight', None)
    if plot_kwargs['highlight'] is not None:
        plot_kwargs['highlight_fromSorted'] = plot_kwargs.get(
            'fighighlight_fromSortedsize', False)
        plot_kwargs['highlight_printNames'] = plot_kwargs.get(
            'highlight_printNames', False)
        plot_kwargs['highlight_fmt'] = plot_kwargs.get('highlight_fmt', 's')
        plot_kwargs['highlight_ms'] = plot_kwargs.get('highlight_ms', 6)
        plot_kwargs['highlight_mfc'] = plot_kwargs.get('highlight_mfc', 'r')
        plot_kwargs['highlight_ecolor'] = plot_kwargs.get(
            'highlight_ecolor', 'pink')
        plot_kwargs['highlight_elinewidth'] = plot_kwargs.get('figsize', 3)

    m = m.copy()
    if not inCols:
        m = m.T
    n_cols = m.shape[1]

    if newFig:
        plt.figure(figsize=plot_kwargs['figsize'])

    if plot_kwargs['highlight'] is not None:
        if isinstance(plot_kwargs['highlight'], str):
            if 'max' in plot_kwargs['highlight']:
                try:
                    n_pick = int(plot_kwargs['highlight'].rsplit('max')[1])
                except:
                    logger.error(
                        "wrong input in highlight argument:" +
                        plot_kwargs['highlight'])
                    raise
                select = np.arange(n_cols-n_pick, n_cols)
                plot_kwargs['highlight'] = m.std(axis=0).argsort()[select]
                # plot_kwargs['highlight'] = m.mean(axis=0).argsort()[select]
            elif 'min' in plot_kwargs['highlight']:
                try:
                    n_pick = int(plot_kwargs['highlight'].rsplit('min')[1])
                except:
                    logger.error(
                        "wrong input in highlight argument:" +
                        plot_kwargs['highlight'])
                    raise
                select = np.arange(n_pick)
                plot_kwargs['highlight'] = m.std(axis=0).argsort()[select]
                # plot_kwargs['highlight'] = m.mean(axis=0).argsort()[select]
            else:
                logger.error(
                    "wrong input in highlight argument:" +
                    plot_kwargs['highlight'])

    if toSort:
        sort_idx = m.std(axis=0).argsort()
        # sort_idx = m.mean(axis=0).argsort()
        m = m[:, sort_idx]

    col_pos = np.arange(n_cols)
    col_mean = m.mean(axis=0)
    col_std = m.std(axis=0)
    plt.errorbar(
        col_pos, col_mean, xerr=0, yerr=col_std, fmt=plot_kwargs['fmt'],
        ms=plot_kwargs['ms'], mfc=plot_kwargs['mfc'],
        ecolor=plot_kwargs['ecolor'], elinewidth=plot_kwargs['elinewidth'])

    if plot_kwargs['highlight'] is not None:
        highlight = plot_kwargs['highlight']
        if toSort and (not plot_kwargs['highlight_fromSorted']):
            try:
                highlight = sort_idx.argsort()[highlight]
            except:
                if isinstance(highlight, list):
                    highlight = np.array(highlight)
                    highlight = sort_idx.argsort()[highlight]
                else:
                    raise
        col_pos = col_pos[highlight]
        col_mean = col_mean[highlight]
        col_std = col_std[highlight]
        plt.errorbar(
            col_pos, col_mean, xerr=0, yerr=col_std,
            fmt=plot_kwargs['highlight_fmt'], ms=plot_kwargs['highlight_ms'],
            mfc=plot_kwargs['highlight_mfc'],
            ecolor=plot_kwargs['highlight_ecolor'],
            elinewidth=plot_kwargs['highlight_elinewidth'])

    if plot_kwargs['draw_axhline']:
        if plot_kwargs['set_axhline'] is None:
            plt.axhline(
                y=m.mean(), color=plot_kwargs['axhline_color'],
                linestyle=plot_kwargs['axhline_linestyle'],
                linewidth=plot_kwargs['axhline_linewidth'])
        else:
            plt.axhline(
                y=plot_kwargs['set_axhline'],
                color=plot_kwargs['axhline_color'],
                linestyle=plot_kwargs['axhline_linestyle'],
                linewidth=plot_kwargs['axhline_linewidth'])
    plt.xlim(-1, n_cols+1)

    if (
            plot_kwargs['highlight'] is not None and
            plot_kwargs['highlight_printNames']):
        plt.xticks(
            col_pos, np.arange(n_cols)[plot_kwargs['highlight']].astype(str),
            rotation=0)
    else:
        if toSort:
            plt.xticks([], [])
        else:
            if plot_kwargs['xticks'] is not None:
                plt.xticks(
                    plot_kwargs['xticks'][0], plot_kwargs['xticks'][1],
                    rotation=plot_kwargs['xticks'][2])

    if plot_kwargs['ylimits'] is not None:
        plt.ylim(plot_kwargs['ylimits'][0], plot_kwargs['ylimits'][1])
    # plt.show()

    return plot_kwargs['highlight']


def plot_adjRI(
        adjRI_df, run_names_df, choose_runs=None, return_df=False,
        palettes=None, cmap='coolwarm', annot=False, figsize=(10, 10),
        vmin=-1, mytitle='', extra_color=None, as_cont=0,
        noCluster=False, noOriginal=False, plot_palette=True):
    if choose_runs is not None:
        adjRI_df = adjRI_df.loc[choose_runs, choose_runs]
        run_names_df = run_names_df.loc[choose_runs]
    if palettes is None:
        palettes = ['tab20', 'winter', 'autumn', '']

    method_colors = get_vector_colors(
        run_names_df['method'], palettes[0],
        plot_palette=plot_palette, size_scale=1)
    k_colors = get_vector_colors(
        run_names_df['k'], palettes[1],
        plot_palette=plot_palette, size_scale=0.5)
    p_colors = get_vector_colors(
        run_names_df['p'], palettes[2],
        plot_palette=plot_palette, size_scale=1)
    n_cluster_colors = get_vector_colors(
        run_names_df['n_clusters'], palettes[3],
        plot_palette=plot_palette, size_scale=0.25)

    if extra_color is not None:
        # extra_color=pd.Series(extra_color, index = adjRI_df.index)
        if len(palettes) > 3:
            extra_palette = palettes[4]
        else:
            extra_palette = 'Blues'

        extra_color = get_vector_colors(
            extra_color, extra_palette,
            plot_palette=plot_palette, as_cont=as_cont)
        run_colors = pd.concat(
            [method_colors, k_colors, p_colors, n_cluster_colors, extra_color],
            axis=1)
    else:
        run_colors = pd.concat(
            [method_colors, k_colors, p_colors, n_cluster_colors],
            axis=1)

    if not noOriginal:
        sns.clustermap(
            adjRI_df, row_cluster=False, col_cluster=False, annot=annot,
            figsize=figsize, vmin=vmin, vmax=1, cmap=cmap,
            row_colors=run_colors, col_colors=run_colors,
            xticklabels=False, yticklabels=False)
        plt.suptitle(mytitle, fontsize=18)
        plt.show()
    if not noCluster:
        sns.clustermap(
            adjRI_df, row_cluster=True,  col_cluster=True,  annot=annot,
            figsize=figsize, vmin=vmin, vmax=1, cmap=cmap,
            row_colors=run_colors, col_colors=run_colors,
            xticklabels=False, yticklabels=False)
        plt.suptitle(mytitle, fontsize=18)
        plt.show()

    if return_df:
        return adjRI_df


def getRGB(values, palette="tab10", return_palette=False):
    nu = np.unique(values.dropna())
    if values.isnull().any():
        nu = nu + 1
    if '_r' in palette:
        palette = palette.rsplit('_r')[0]
        mycolor_palette = sns.color_palette(palette, len(nu))
        mycolor_palette.reverse()
    else:
        mycolor_palette = sns.color_palette(palette, len(nu))
    # sns.palplot(mycolor_palette)
    lut = dict(zip(nu, mycolor_palette))
    rgb = values.map(lut)

    if return_palette:
        return rgb, mycolor_palette
    else:
        return rgb


def takespread(sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]


def get_vector_colors(
        vector, palette, plot_palette=True, as_cont=0,
        return_palette=False, size_scale=1):
    if not isinstance(vector, pd.Series):
        vector = pd.Series(vector, index=vector)
    vector_unique = np.unique(vector.dropna())
    nu = len(vector_unique)
    if vector.isnull().any():
        nu = nu + 1
        vector_unique = np.append(vector_unique, np.nan)

    if nu > 1:
        vector_colors, vector_palette = getRGB(
            vector, palette=palette, return_palette=True)
        if as_cont > 0:
            if plot_palette:
                sns.palplot(
                    np.array(vector_palette)[
                        list(takespread(np.arange(nu), as_cont))],
                    size=size_scale)
                plt.xticks(
                    range(as_cont),
                    list(takespread(vector_unique, as_cont)),
                    rotation=90)
                plt.title(vector.name)
                plt.show()
        else:
            if plot_palette:
                sns.palplot(vector_palette, size=size_scale)
                plt.xticks(range(nu), vector_unique, rotation=90)
                plt.title(vector.name)
                plt.show()
    else:
        vector_colors = pd.Series(
            np.zeros((vector.shape[0],), dtype=object), index=vector.index)
        for row in vector_colors.index:
            vector_colors.loc[row] = (1, 1, 1)

    if return_palette:
        return vector_colors, vector_palette
    else:
        return vector_colors


def plot_projection(
        choose_mask, clusters, cluster_col_name, proj, s=100,
        other_color=None, toSave=False, main_name="gene",
        fpath='_projected_to_clusters.png', cmap="tab10", axes_lim=None):
    n = len(clusters[cluster_col_name].unique())

    if other_color is None:
        mycolor_palette = sns.color_palette(cmap, n)
        max_color = choose_mask.max(axis=1)
    else:
        max_color = other_color.copy()

    if axes_lim is None:
        axesLim = choose_mask.max(axis=0)
        axesLim = np.round(axesLim.values, 2)
    else:
        axesLim = np.repeat(axes_lim, n)

    sns.set_style('whitegrid')
    all_markers = [
        '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's',
        'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']

    # Project each patient/gene to 3 atom communitities
    markers = np.array(all_markers[:n])
    names = np.array(['c'+str(s) for s in range(n)])
    names = np.append(names, 'ALL')

    if isinstance(proj, list):
        proj = np.array(proj)

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')
    ax = f.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(
        '{} {} projected on \natom communities {}'.format(
            names[n], main_name, names[proj]), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.set_xlabel('C'+str(proj[0]), fontsize=16)
    ax.set_xlim3d(0, axesLim[proj[0]])
    ax.set_ylabel('C'+str(proj[1]), fontsize=16)
    ax.set_ylim3d(0, axesLim[proj[1]])
    ax.set_zlabel('C'+str(proj[2]), fontsize=16)
    ax.set_zlim3d(0, axesLim[proj[2]])
    for i in np.arange(0, n):
        if other_color is None:
            single_cmap = LinearSegmentedColormap.from_list(
                "", [(1.0, 1.0, 1.0), mycolor_palette[i]], N=100)
        else:
            single_cmap = cmap

        com = clusters.index[(clusters[cluster_col_name] == i)]
        xyz = choose_mask.loc[com]
        color = max_color.loc[com]
        ax.scatter(
            xyz.iloc[:, proj[0]], xyz.iloc[:, proj[1]], xyz.iloc[:, proj[2]],
            c=color, marker='o', s=s, cmap=single_cmap)

    if toSave:
        f.savefig(
            fpath, transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close('all')
    sns.set_style('white')


def plot_cluster_projection(
        choose_mask, clusters, cluster_col_name, proj, s=100, other_color=None,
        toSave=False, main_name="gene", fpath='_projected_to_clusters.png',
        cmap="tab10", axes_lim=None):
    # For each set of style and range settings, plot n random points in the box
    n = len(clusters[cluster_col_name].unique())

    if other_color is None:
        mycolor_palette = sns.color_palette(cmap, n)
        max_color = choose_mask.max(axis=1)
    else:
        max_color = other_color.copy()

    if axes_lim is None:
        axesLim = choose_mask.max(axis=0)
        axesLim = np.round(axesLim.values, 2)
    else:
        axesLim = np.repeat(axes_lim, n)

    sns.set_style('whitegrid')
    all_markers = [
        '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's',
        'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']

    # Project each patient/gene to 3 atom communitities
    markers = np.array(all_markers[:n])
    names = np.array(['c'+str(s) for s in range(n)])
    names = np.append(names, 'ALL')

    if isinstance(proj, list):
        proj = np.array(proj)
    js = [1, 2, 3]
    xs = [0, 0, 1]
    ys = [1, 2, 2]

    # Patients/Genes themselves are grouped to one of these communities
    # according to which has the max coef value
    # but they still have values to the rest of the communities
    f, ax = plt.subplots(n+1, 4, figsize=(32, (n+1)*8))
    # add and set for main subplot
    # where we plot all patients/genes (from all communities)
    # and project them to the 3 pre-selected atom communities

    ax[n, 0].axis('off')
    ax[n, 0] = f.add_subplot(n+1, 4, n+(n*3)+1, projection='3d')
    ax[n, 0].set_title(
        '{} {} cluster(s) projected on \natom communities {}'.format(
            names[n], main_name, names[proj]), fontsize=16)
    ax[n, 0].tick_params(axis='both', which='major', labelsize=10)
    ax[n, 0].tick_params(axis='both', which='minor', labelsize=10)
    ax[n, 0].set_xlabel('C'+str(proj[0]), fontsize=16)
    ax[n, 0].set_xlim3d(0, axesLim[proj[0]])
    ax[n, 0].set_ylabel('C'+str(proj[1]), fontsize=16)
    ax[n, 0].set_ylim3d(0, axesLim[proj[1]])
    ax[n, 0].set_zlabel('C'+str(proj[2]), fontsize=16)
    ax[n, 0].set_zlim3d(0, axesLim[proj[2]])

    for j in js:
        x_ = xs[j-1]
        y_ = ys[j-1]
        ax[n, j].axis('off')
        ax[n, j] = f.add_subplot(n+1, 4, n+(n*3)+1+j)
        ax[n, j].set_title(
            '{} {} cluster(s) projected on \natom communities {}'.format(
                names[n], main_name, names[proj[[x_, y_]]]), fontsize=16)
        ax[n, j].tick_params(axis='both', which='major', labelsize=10)
        ax[n, j].tick_params(axis='both', which='minor', labelsize=10)
        ax[n, j].set_xlabel('C'+str(proj[x_]), fontsize=16)
        ax[n, j].set_xlim(0, axesLim[proj[x_]])
        ax[n, j].set_ylabel('C'+str(proj[y_]), fontsize=16)
        ax[n, j].set_ylim(0, axesLim[proj[y_]])

    # add and set subplots for each patient/gene community
    # (according to max atom community)
    for i in np.arange(0, n):
        if other_color is None:
            single_cmap = LinearSegmentedColormap.from_list(
                "", [(1.0, 1.0, 1.0), mycolor_palette[i]], N=100)
        else:
            single_cmap = cmap
        com = clusters.index[(clusters[cluster_col_name] == i)]
        xyz = choose_mask.loc[com]
        color = max_color.loc[com]

        # The main plot
        ax[n, 0].scatter(
            xyz.iloc[:, proj[0]], xyz.iloc[:, proj[1]], xyz.iloc[:, proj[2]],
            c=color, marker=markers[i], s=s, cmap=single_cmap)

        # The patient/gene community plot on 3d projection
        ax[i, 0].axis('off')
        ax[i, 0] = f.add_subplot(n+1, 4, i+(i*3)+1, projection='3d')
        ax[i, 0].set_title(
            '{} {} cluster(s) projected on \natom communities {}'.format(
                names[i], main_name, names[proj]), fontsize=16)
        ax[i, 0].tick_params(axis='both', which='major', labelsize=10)
        ax[i, 0].tick_params(axis='both', which='minor', labelsize=10)
        ax[i, 0].set_xlabel('C'+str(proj[0]), fontsize=16)
        ax[i, 0].set_xlim3d(0, axesLim[proj[0]])
        ax[i, 0].set_ylabel('C'+str(proj[1]), fontsize=16)
        ax[i, 0].set_ylim3d(0, axesLim[proj[1]])
        ax[i, 0].set_zlabel('C'+str(proj[2]), fontsize=16)
        ax[i, 0].set_zlim3d(0, axesLim[proj[2]])
        ax[i, 0].scatter(
            xyz.iloc[:, proj[0]], xyz.iloc[:, proj[1]], xyz.iloc[:, proj[2]],
            c=color, marker=markers[i], s=s, cmap=single_cmap)

        # The patient/gene community plot on 2d projection (1st and 2nd C)
        for j in js:
            x_ = xs[j-1]
            y_ = ys[j-1]
            ax[i, j].axis('off')
            ax[i, j] = f.add_subplot(n+1, 4, i+(i*3)+1+j)
            ax[i, j].set_title(
                '{} {} cluster(s) projected on \natom communities {}'.format(
                    names[i], main_name, names[proj[[x_, y_]]]), fontsize=16)
            ax[i, j].tick_params(axis='both', which='major', labelsize=10)
            ax[i, j].tick_params(axis='both', which='minor', labelsize=10)
            ax[i, j].set_xlabel('C'+str(proj[x_]), fontsize=16)
            ax[i, j].set_xlim(0, axesLim[proj[x_]])
            ax[i, j].set_ylabel('C'+str(proj[y_]), fontsize=16)
            ax[i, j].set_ylim(0, axesLim[proj[y_]])

            ax[i, j].scatter(
                xyz.iloc[:, proj[x_]], xyz.iloc[:, proj[y_]],
                c=color, marker=markers[i], s=s, cmap=single_cmap)

            ax[n, j].scatter(
                xyz.iloc[:, proj[x_]], xyz.iloc[:, proj[y_]],
                c=color, marker=markers[i], s=s, cmap=single_cmap)

    if toSave:
        f.savefig(
            fpath, transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close('all')
    sns.set_style('white')


def plot_errorbars_cox(
        cox_summary, fig_count, saveReport, report_outdir, vline=1,
        sort_col='exp(coef)', m_col='exp(coef)',
        low_col='exp(lower_95)', up_col='exp(upper_95)', p_col='p_fdr_bh',
        mytitle='signature markers from univariate Cox regression',
        figsize=(8, 8)):
    summary2plot = cox_summary.copy()
    summary2plot = summary2plot.sort_values(by=[sort_col], ascending=True)
    starred_names = []
    for name in summary2plot.index.values:
        new_name = name
        if cox_summary[p_col][name] < 0.05:
            new_name = new_name+'*'
            if cox_summary[p_col][name] < 0.01:
                new_name = new_name+'*'
                if cox_summary[p_col][name] < 0.001:
                    new_name = new_name+'*'
        starred_names.append(new_name)

    if saveReport:
        fig_count += 1
        logger.info('[Figure '+str(fig_count)+']')
    if True:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax2 = ax.twinx()

        avgs = summary2plot[m_col]
        xerr_low = summary2plot[low_col]
        xerr_up = summary2plot[up_col]

        # ad-hoc string join solution
        a1 = avgs.values.round(2).astype(str).tolist()
        a2 = xerr_low.values.round(2).astype(str).tolist()
        a3 = xerr_up.values.round(2).astype(str).tolist()
        a_empty = ['' for i in range(len(a1))]

        str_j = map(' ('.join, list(zip(*[a1, a2])))
        str_j = map(', '.join, list(zip(*[str_j, a3])))
        str_j = map(')'.join, list(zip(*[str_j, a_empty])))

        str_yticklabels = list(str_j)
        ax.errorbar(
            avgs, range(summary2plot.shape[0]),
            xerr=[avgs - xerr_low, xerr_up - avgs],
            color='r', fmt='s', capsize=5, ecolor='k')
        ax2.errorbar(
            avgs, range(summary2plot.shape[0]),
            xerr=[avgs - xerr_low, xerr_up - avgs],
            color='r', fmt='s', capsize=5, ecolor='k')

        if vline is not None:
            ax.axvline(x=vline, color='k', linestyle='--')

        ax.set_yticks(range(summary2plot.shape[0]))
        ax.set_yticklabels(starred_names)

        ax2.set_yticks(range(summary2plot.shape[0]))
        ax2.set_yticklabels(str_yticklabels)

        ax.set_xlabel(m_col)
        plt.title(mytitle+'\n(*** p<0.001, ** p<0.01, * p<0.05)')
    if saveReport:
        plt.savefig(
            report_outdir+'Figure'+str(fig_count)+'.png',
            transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
    if not saveReport:
        plt.show()
    else:
        plt.close("all")

    return fig_count


def plot_knn_network(
        similarity, data_to_cluster, layout_mode, pos_param,
        inference_run_id, node_color_settings, saveimg, img_outDir,
        imgCount_basic, resolution_extention):

    imgCount_basic += 1
    imgCount = 0
    # PLOT similarity #
    myfig = sns.clustermap(
        similarity.A, col_cluster=True, row_cluster=True,
        xticklabels=True, yticklabels=True, square=True,
        cmap='Reds', figsize=(6, 6))
    if saveimg:
        imgCount += 1
        myfig.savefig(
            (
                img_outDir+str(imgCount_basic)+'__' +
                inference_run_id+'__'+str(imgCount)+resolution_extention),
            transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close('all')
    else:
        plt.show()

    # DRAW  NETWORK #
    text = 'patients'
    size_const = 2000  # some constant for the nodes size
    edge_line_width = -5  # some constant for the edges width
    nodeSize = abs(data_to_cluster).sum(axis=1)  # size
    nodeSize = nodeSize/nodeSize.max()
    # nodeColor = patient_grade_group.copy() ## color
    # nodeCmap = 'cat_cont'

    mytitle = inference_run_id+'_pos_param_'+str(pos_param)
    for nodeColor, nodeCmap in node_color_settings:
        myfig, net_pos, G, _ = networkx_draw_network(
            similarity, nodeSize.values, nodeColor.values,
            size_const=size_const,
            which_nodes=None, all_edges=False,
            layout=layout_mode, specified_pos=None, pos_param=pos_param,
            myfigsize=(12, 10), mytitle=mytitle, print_node_names=False,
            font_size=20,
            node_line_width=1, edge_line_width=edge_line_width,
            nodeCmap=nodeCmap, nodePalette=cl_palette,
            mySeed=8)
        if saveimg:
            imgCount += 1
            myfig.savefig(
                (
                    img_outDir+str(imgCount_basic)+'__'+inference_run_id +
                    '__'+str(imgCount)+resolution_extention),
                transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
            plt.close('all')
        else:
            plt.show()

    return imgCount_basic, imgCount
