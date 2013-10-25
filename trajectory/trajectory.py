# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:14:47 2013

@author: mclaus
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def trajectory(axes, x, y, dim=1, narrs=30, dspace=0.5, direc='pos', \
                          hl=0.3, hw=6, c='black', latlon=False):
    ''' dim    :  ID dimension of input array

        narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head

        hw     :  width of the arrow head

        c      :  color of the edge and face of the arrow head
    '''
    # Convert to mask array
    x = np.ma.asarray(x)
    y = np.ma.asarray(y)

    if x.shape != y.shape:
        raise "Inconsistend shape. x and y must have the same shape."

    if x.ndim < 1 or x.ndim > 2:
        raise "Input arrays x and y must be of rank 1 or 2."

    if np.rank(x) == 1:
        # add singleton dimension to the end
        x = x[..., np.newaxis]
        y = y[..., np.newaxis]
        dim = 1

    if dim == 0:
        # transpose to get ID dimension to the end
        x = x.T
        y = y.T
        dim = 1

    # mask all missig values
    np.ma.masked_where(y.mask, x, copy=False)
    np.ma.masked_where(x.mask, y, copy=False)

    for t in xrange(x.shape[dim]):
        xt = x[..., t].compressed()
        yt = y[..., t].compressed()
        arrow_plot(axes, xt, yt, latlon, narrs, dspace, direc, hl, hw, c)

#    axes.set_xlim(x.min() * .9, x.max() * 1.1)
#    axes.set_ylim(y.min() * .9, y.max() * 1.1)


def arrow_plot(axes, x, y, latlon=False, narrs=30, dspace=0.5, direc='pos', \
                          hl=0.3, hw=6, c='black'):
    ''' narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head

        hw     :  width of the arrow head

        c      :  color of the edge and face of the arrow head
    '''

    # r is the distance spanned between pairs of points
    if len(x) == 1:
        return

    xscale = np.max(x) - np.min(x)
    yscale = np.max(y) - np.min(y)

    r = [0]
    rtot = [0]
    for i in range(1, len(x)):
        dx = (x[i] - x[i - 1]) / xscale
        dy = (y[i] - y[i - 1]) / yscale
        r.append(np.sqrt(dx * dx + dy * dy))
        rtot.append(rtot[-1] + r[-1])
    r = np.array(r)
    rtot = np.array(rtot)

    # based on narrs set the arrow spacing
    aspace = rtot[-1] / narrs

    if direc is 'neg':
        dspace = -1. * abs(dspace) * aspace
    else:
        dspace = abs(dspace) * aspace

    arrowPos = np.linspace(0, rtot[-1], narrs + 1) + dspace
    arrowPos.sort()

    idx = 0
    for arr in range(narrs + 1):
        # get index of the starting point of the line segment to
        # put the arrow head on
        for idx in range(idx, len(rtot) - 1)[:]:
            if rtot[idx] <= arrowPos[arr] and rtot[idx + 1] >= arrowPos[arr]:
                idx1 = idx
                idx2 = idx1 + 1
                break
            else:
                idx1 = -1
        if idx1 < 0:  # no linesegment found
            break
        x1, x2 = x[idx1], x[idx2]
        y1, y2 = y[idx1], y[idx2]
        ds = arrowPos[arr] - rtot[idx1]
        theta = np.arctan2((x2 - x1), (y2 - y1))
        a0 = [np.sin(theta) * (ds + hl / 2.) + x1,
              np.cos(theta) * (ds + hl / 2.) + y1]
        a1 = [np.sin(theta) * (ds - hl / 2.) + x1,
              np.cos(theta) * (ds - hl / 2.) + y1]

        if direc is 'neg':
            a0, a1 = a1, a0

        if isinstance(axes, Basemap):
            (a0[0], a1[0]), (a0[1], a1[1]) = axes([a0[0], a1[0]],
                                                  [a0[1], a1[1]])
            if not axes.ax:
                loc_ax = plt.gca()
        else:
            loc_ax = axes

        loc_ax.annotate('', xy=a0, xycoords='data',
                xytext=a1, textcoords='data',
                arrowprops=dict(headwidth=hw, frac=1., ec=c, fc=c))

    if isinstance(axes, Basemap):
        axes.plot(x, y, color=c, latlon=latlon)
    else:
        axes.plot(x, y, color=c)
