import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def plot_loss(x, y, xlabel, ylabel, title):
    print('away from diagonal portion:', np.mean(x != y))

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # project large values back to reasonable range for plotting purpose
    # set axis limits
    xlim = min(1.1, max(1.05, max(x)))
    ylim = xlim
    axScatter.set_xlim((-0.05, xlim))
    axScatter.set_ylim((-0.05, ylim))
    x[x >= xlim] = xlim - 0.01
    y[y >= ylim] = ylim - 0.01

    # the scatter plot:
    axScatter.scatter(x, y, color='b', alpha=0.3)
    axScatter.plot([0, 1], [0, 1], ls='--', color='g')
    axScatter.set_xlabel(xlabel, fontdict={'fontsize': 12})
    axScatter.set_ylabel(ylabel, fontdict={'fontsize': 12})

    # now determine nice limits by hand:
    # binwidth = 0.25
    # xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    # lim = (int(xymax/binwidth) + 1) * binwidth

    # axScatter.set_xlim((-lim, lim))
    # axScatter.set_ylim((-lim, lim))
    # bins = np.arange(-lim, lim + binwidth, binwidth)

    axHistx.hist(x, bins=10, color='g', alpha=0.3)
    axHistx.set_yscale('log')
    axHisty.hist(y, bins=10, color='g', alpha=0.3, orientation='horizontal')
    axHisty.set_xscale('log')

    # axHistx.set_xlim(axScatter.get_xlim())
    # axHisty.set_ylim(axScatter.get_ylim())

    axHistx.set_title(title, fontdict={'fontsize': 15}, loc='center')
