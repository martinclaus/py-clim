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
    joint_mask = np.logical_or(x.mask, y.mask)
    x[joint_mask] = np.ma.masked
    y[joint_mask] = np.ma.masked

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
    r = [0]
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        r.append(np.sqrt(dx * dx + dy * dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    # based on narrs set the arrow spacing
    aspace = r.sum() / narrs

    if direc is 'neg':
        dspace = -1. * abs(dspace)
    else:
        dspace = abs(dspace)

    arrowData = []  # will hold tuples of x,y,theta for each arrow
    arrowPos = aspace * (dspace)  # current point on walk along data
                                 # could set arrowPos to 0 if you want
                                 # an arrow at the beginning of the curve

    ndrawn = 0
    rcount = 1
    while arrowPos < r.sum() and ndrawn < narrs:
        x1, x2 = x[rcount - 1], x[rcount]
        y1, y2 = y[rcount - 1], y[rcount]
        da = arrowPos - rtot[rcount]
        theta = np.arctan2((x2 - x1), (y2 - y1))
        ax = np.sin(theta) * da + x1
        ay = np.cos(theta) * da + y1
        arrowData.append((ax, ay, theta))
        ndrawn += 1
        arrowPos += aspace
        while arrowPos > rtot[rcount + 1]:
            rcount += 1
            if arrowPos > rtot[-1]:
                break

    # could be done in above block if you want
    for ax, ay, theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta) * hl / 2. + ax
        dy0 = np.cos(theta) * hl / 2. + ay
        dx1 = -1. * np.sin(theta) * hl / 2. + ax
        dy1 = -1. * np.cos(theta) * hl / 2. + ay

        if direc is 'pos':
            ax0 = dx0
            ay0 = dy0
            ax1 = dx1
            ay1 = dy1
        else:
            ax0 = dx1
            ay0 = dy1
            ax1 = dx0
            ay1 = dy0

        if isinstance(axes, Basemap):
            (ax0, ax1), (ay0, ay1) = axes([ax0, ax1], [ay0, ay1])
            if not axes.ax:
                loc_ax = plt.gca()
        else:
            loc_ax = axes

        loc_ax.annotate('', xy=(ax0, ay0), xycoords='data',
                xytext=(ax1, ay1), textcoords='data',
                arrowprops=dict(headwidth=hw, frac=1., ec=c, fc=c))

    if isinstance(axes, Basemap):
        axes.plot(x, y, color=c, latlon=latlon)
    else:
        axes.plot(x, y, color=c)
