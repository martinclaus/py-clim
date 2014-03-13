# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:55:49 2014

@author: mclaus

Collection of useful functions to adjust plots produced by PYNGL, the python
binding to the NCAR Graphics Library.
"""

import Ngl as ngl
import numpy as np
import copy


def map_data_to_color(wks, data, c_offset=3):
    ''' Returns the color indices that best match the data array

    color_indices = map_data_to_color(wks, data, c_offset=3)

    wks :     The identifier returned from calling Ngl.open_wks.

    data :    Numpy array or python list of data

    c_offset: number of colors to omitt at the beginnig of the colormap, like
              foreground and background color.
    '''
    c_offset = int(c_offset)
    ncolors = ngl.retrieve_colormap(wks).shape[0] - 2 - c_offset
    dmin = np.min(data).astype(float)
    data_norm = (data - dmin) / np.max(data - dmin)
    return (data_norm * ncolors).astype(int) + c_offset


def get_cyclic_colormap(ncolor=180., saturation=1., value=.75,
                           bg_color=(1., 1., 1.), fg_color=(0., 0., 0.)):
    ''' Returns a cyclic colormap by conversion from the HSV to RGB color space

    cmap = get_cyclic_colormap(ncolor=180., saturation=1., value=.75,
                               bg_color=(1., 1., 1.), fg_color=(0., 0., 0.))
    '''
    hue = np.linspace(0., 360., ncolor, endpoint=False)
    cmap = np.zeros((ncolor + 2, 3))
    cmap[0, :] = bg_color
    cmap[1, :] = fg_color
    for i, h in enumerate(hue):
        cmap[i + 2, :] = ngl.hsvrgb(h, saturation, value)
    return cmap


def add_panel_label(wks, plot, panLabel="", factor=(1., 1.),
                    placement="ul", axfunc=min, rlist=None):
    vpXF, vpYF, vpWidthF, vpHeightF = [ngl.get_float(plot, name) for name in
                                       ["vpXF", "vpYF",
                                       "vpWidthF", "vpHeightF"]]
    length = .125 * axfunc([vpWidthF, vpHeightF])
    lengthX = length * factor[0]
    lengthY = length * factor[1]
    margin = .25 * min((lengthX, lengthY))
    print vpXF, vpYF, vpWidthF, vpHeightF

    ul_edge = [vpXF + .5 * vpWidthF, vpYF + .5 * vpHeightF]
    if placement[0] == "u":
        ul_edge[1] = vpYF - margin
    elif placement[0] == "l":
        ul_edge[1] = vpYF - vpHeightF + margin + lengthY
    if placement[1] == "l":
        ul_edge[0] = vpXF + margin
    elif placement[1] == "r":
        ul_edge[0] = vpXF + vpWidthF - margin - lengthX
    if len(ul_edge) != 2:
        raise ValueError("placement is not in ul, ur, ll, lr")

    x_box = ul_edge[0] + np.array((0., 0., lengthX, lengthX, 0.))
    y_box = ul_edge[1] - np.array((0., lengthY, lengthY, 0., 0.))

    box_res = {"gsFillColor": 0,
               "gsEdgeColor": 1,
               "gsEdgesOn": True}

    tx_res = {"txFontHeightF": .02}

    x_tx = ul_edge[0] + .5 * lengthX
    y_tx = ul_edge[1] - .5 * lengthY

    if rlist:
        for key, value in rlist.__dict__:
            if key[0:2] == "gs" and key[2].isupper():
                box_res[key] = value
            elif key[0:2] == "tx" and key[2].isupper():
                tx_res[key] = value

    ngl.polygon_ndc(wks, x_box, y_box, box_res)
    ngl.text_ndc(wks, panLabel, x_tx, y_tx, tx_res)


def set_plot_position(plot, vpXY=(.2, .8), vpWH=(.6, .6)):
    rlist = {}
    rlist["vpXF"], rlist["vpYF"] = vpXY
    rlist["vpWidthF"], rlist["vpHeightF"] = vpWH
    _set_values(plot, rlist)


def cn_sym_min_max(x, res, ncontours=15, outside=True, aboutZero=True):
    min_x = np.min(x)
    max_x = np.max(x)
    if min_x == max_x:
        min_x, max_x = -1., 1.
    min_out, max_out, step_size = ngl.nice_cntr_levels(min_x, max_x,
                                                       outside=outside,
                                                       max_steps=ncontours,
                                                       aboutZero=aboutZero)
    newres = {"cnLevelSelectionMode": "ManualLevels",
              "cnMinLevelValF": min_out,
              "cnMaxLevelValF": max_out,
              "cnLevelSpacingF": (max_out - min_out) / step_size}
    res.__dict__.update(newres)


def cn_neg_dash_contours(contour):
    levels = ngl.get_float_array(contour, "cnLevels")

    patterns = np.zeros((len(levels)), 'i')
    patterns[:] = 0     # solid line
    for i in xrange(len(levels)):
        if (levels[i] < 0.):
            patterns[i] = 2
    rlist = {"cnLineDashPatterns": patterns,
             "cnMonoLineDashPattern": False}
    _set_values(contour, rlist)


def cn_thick_zero_contour(contour, factor=2.):
    cn_line_thickness(contour, val=0., factor=factor)


def cn_line_thickness(contour, val=0., factor=2.):
    cnLevels = ngl.get_float_array(contour, "cnLevels")
    cnLineThicknesses = ngl.get_float_array(contour, "cnLineThicknesses")

    for i, thickness in enumerate(cnLineThicknesses[:]):
        if cnLevels[i] in val:
            cnLineThicknesses[i] = thickness * factor

    rlist = {"cnMonoLineThickness": False,
             "cnLineThicknesses": cnLineThicknesses}
    _set_values(contour, rlist)


def tm_lat_lon_labels(contour, ax="XB", direction="zonal"):
    deg_c = '~S~o~N~'
    if direction[0].lower() == "z":
        hem_label = (deg_c + "W", deg_c + "E")
    elif direction[0].lower() == "m":
        hem_label = (deg_c + "S", deg_c + "N")
    tm_values = ngl.get_float_array(contour, "tm" + ax + "Values")
    mtm_values = ngl.get_float_array(contour, "tm" + ax + "MinorValues")

    rlist = {"tm" + ax + "Mode": "Explicit",
             "tm" + ax + "Values": tm_values,
             "tm" + ax + "MinorValues": mtm_values,
             "tm" + ax + "Labels": ["%d" % abs(val) + hem_label[0] if val < 0.
                                    else "%d" % abs(val) + hem_label[1]
                                    for val in tm_values]}
    _set_values(contour, rlist)


def tm_deg_labels(plt_obj, ax="YL"):
    tm_values = ngl.get_float_array(plt_obj, "tm" + ax + "Values")
    mtm_values = ngl.get_float_array(plt_obj, "tm" + ax + "MinorValues")

    labels = ["%3.1f~S~o~N~" % val for val in tm_values]

    rlist = {"tm" + ax + "Mode": "Explicit",
             "tm" + ax + "Values": tm_values,
             "tm" + ax + "MinorValues": mtm_values,
             "tm" + ax + "Labels": labels}

    _set_values(plt_obj, rlist)


def lb_create_labelbar(wks, vpXY, vpWH, nboxes=11, levels=(-1., 1.),
                       frm_str="{}", rlist=None):
    # add label bar for depth
    levels = np.linspace(*levels, num=nboxes)
    labels = [frm_str.format(val) for val in levels]
    labres = ngl.Resources()
    labres.lbAutoManage = False
    labres.vpWidthF, labres.vpHeightF = vpWH
    labres.lbMonoFillPattern = 21
    labres.lbOrientation = "vertical"
    labres.lbFillColors = map_data_to_color(wks, levels)
    labres.lbLabelStride = 10
    if rlist:
        labres.__dict__.update(rlist)
    lb = ngl.labelbar_ndc(wks, nboxes, labels, *vpXY, rlistc=labres)
    return lb


def lb_set_phase_labels(contour):
    lstride = ngl.get_integer(contour, "lbLabelStride")
    label = ngl.get_string_array(contour, "lbLabelStrings")
    fun_code = ngl.get_string(contour, "lbLabelFuncCode")
    pi_str = fun_code.join(["", "F33", "p"])
    pi_half_str = fun_code.join(["", "H2V15F33", "p", "H-15V-1", "_",
                                 "H-16V-30", "2"])
    label[::lstride] = ["-" + pi_str, "-" + pi_half_str, "0",
                        pi_half_str, pi_str]
    rlist = {"cnExplicitLabelBarLabelsOn": True,
             "lbLabelStrings": label}
    _set_values(contour, rlist)


def lb_format_labels(contour, fmt, minmax=(0., 1.)):
    levels = [float(s) for s in
        ngl.get_string_array(contour, "cnLevels")]
    labels = [fmt % lev for lev in levels]
    if ngl.get_string(contour, "cnLabelBarEndStyle") == "IncludeMinMaxLabels":
        minlabel = [fmt % minmax[0]]
        minlabel.extend(labels)
        labels = minlabel
        labels.append(fmt % minmax[1])
    rlist = {"cnExplicitLabelBarLabelsOn": True,
             "lbLabelStrings": labels}
    _set_values(contour, rlist)


def trj(wks, x, y, time, res=None):
    ''' Plots trajectories with arrow heads attached.

    plot = trj(wks, x, y, time, res=None)

    wks : The identifier returned from calling Ngl.open_wks.

    x,y : The X and Y coordinates of the curve(s). These values can be
          one-dimensional NumPy arrays, NumPy masked arrays or two-dimensional
          NumPy arrays. If x and/or y are two-dimensional, then the leftmost
          dimension determines the number of curves.

    time : Time coordinates of the trajectories. These values can be
          one-dimensional NumPy arrays, NumPy masked arrays

    res:   Resource list using following special resources:

    Special resources:

    trjArrowStep  :     Number of samples between arrow heads.
                        Default: 10

    trjArrowOffsetF :   Shift the position of the arrows along the curve.
                        Should be between 0. and 1.
                        Default: 0.5

    trjArrowDirection : can be 'pos' or 'neg' to select direction of the
                        arrows.
                        Default: 'pos'

    trjArrowXShapeF :   Array containing the x NDC coordinates of the arrow
                        head relative to the heads centre.
                        Default: Equiliteral triangle

    trjArrowYShapeF :   Array containing the y NDC coordinates of the arrow
                        head relative to the heads centre.
                        Default: Equiliteral triangle

    trjArrowXScaleF :   Scales arrow head in X direction, befor rotation is
                        applied.
                        Default: 1.

    trjArrowYScaleF :   Scales arrow head in y direction, befor rotation is
                        applied.
                        Default: 1.

    trjArrowSizeF :     Scales the size of an arrow head.
                        Default: 0.02

    The arrow heads are plotted with add_polygon, so all gs* attributes of the
    resource list are applied to the arrow heads polygon.
    '''

    if not res:
        res = ngl.Resources()

    # set default values:
    if not hasattr(res, 'trjArrowStep'):
        res.trjArrowStep = 10
    if not hasattr(res, 'trjArrowOffsetF'):
        res.trjArrowOffsetF = .5
    else:
        res.trjArrowOffsetF -= np.floor(res.trjArrowOffsetF)
    if not hasattr(res, 'trjArrowDirection'):
        res.trjArrowDirection = 'pos'
    if not hasattr(res, 'trjArrowXShapeF'):
        res.trjArrowXShapeF = [np.sqrt(3) / 3.,
                              -np.sqrt(3) / 6.,
                              -np.sqrt(3) / 6.]
    res.trjArrowXShapeF = np.asarray(res.trjArrowXShapeF)
    if not hasattr(res, 'trjArrowYShapeF'):
        res.trjArrowYShapeF = [0., .5, -.5]
    res.trjArrowYShapeF = np.asarray(res.trjArrowYShapeF)
    if not hasattr(res, 'trjArrowXScaleF'):
        res.trjArrowXScaleF = 1.
    if not hasattr(res, 'trjArrowYScaleF'):
        res.trjArrowYScaleF = 1.
    if not hasattr(res, 'trjArrowSizeF'):
        res.trjArrowSizeF = .02

    # check for draw and frame
    if hasattr(res, "nglDraw"):
        doDraw = res.nglDraw
    else:
        doDraw = True
    res.nglDraw = False
    if hasattr(res, "nglFrame"):
        doFrame = res.nglFrame
    else:
        doFrame = True
    res.nglFrame = False

    # Convert to mask array
    x = np.ma.asarray(x)
    y = np.ma.asarray(y)
    time = np.ma.asarray(time)

    # check input data
    if x.shape != y.shape:
        raise ValueError("Inconsistend shape. x, y and time must have the "
                            + "same shape.")

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError("Input arrays x and y must be of rank 1 or 2.")

    if np.rank(x) == 1:
        # add singleton dimension to the begining
        x = x[np.newaxis, ...]
        y = y[np.newaxis, ...]

    # Dimension of trajectories
    dim = 0

    # mask all missig values
    np.ma.masked_where(y.mask, x, copy=False)
    np.ma.masked_where(x.mask, y, copy=False)

    # create line plot resource
    res_lines = copy.deepcopy(res)

    # remove trj* attributes from res_lines
    for attr in dir(res_lines):
        if attr[0:3] == "trj" or (attr[0:2] == "gs" and attr[2] != "n"):
            delattr(res_lines, attr)

    # create line plot
    plot = ngl.xy(wks, x, y, res_lines)

    # get axes length in data coordinates
    xAxisLen, yAxisLen = [ngl.get_float(plot, "tr" + ax + "MaxF")
                          - ngl.get_float(plot, "tr" + ax + "MinF")
                          for ax in "XY"]

    # get marker color
    # to be implemented

    # place Marker
    marker_id = []
    for t in xrange(x.shape[dim]):
        xt = x[t, ...].compressed()
        yt = y[t, ...].compressed()
        tt = time[::res.trjArrowStep]

        # shift time by offset
        if res.trjArrowOffsetF != 0.:
            tt = tt[:-1] + res.trjArrowOffsetF * (tt[1:] - tt[:-1])

        # itterate over markers
        for tm in tt:
            # find nearest indices in time array
            idx = (np.abs(time - tm)).argmin().min()
            if time[idx] < tm:
                idx1 = idx
                idx2 = idx + 1
            elif time[idx] > tm:
                idx1 = idx - 1
                idx2 = idx
            else:
                if idx == 0:
                    idx1 = idx
                    idx2 = idx + 1
                else:
                    idx1 = idx - 1
                    idx2 = idx

            if idx >= len(xt) - 1:
                continue
            # interpolate linearly to get coordinates
            ds = (tm - time[idx1]) / (time[idx2] - time[idx1])
            xm, ym = [coord[idx1] + ds * (coord[idx2] - coord[idx1])
                      for coord in [xt, yt]]
            x1, y1 = ngl.datatondc(plot, xt[idx1], yt[idx1])
            x2, y2 = ngl.datatondc(plot, xt[idx2], yt[idx2])
            angle = np.arctan2(y2 - y1, x2 - x1)

            # create marker resource
            res_marker = copy.deepcopy(res)
            # scale adjust marker scale
            res_marker.trjArrowXScaleF = res.trjArrowXScaleF * xAxisLen
            res_marker.trjArrowYScaleF = res.trjArrowYScaleF * yAxisLen

            marker_id.append(
                _plot_trj_marker(wks, plot, xm, ym, angle, res_marker))

        if doDraw:
            ngl.draw(plot)
            res.nglDraw = True
        if doFrame:
            ngl.frame(wks)
            res.nglFrame = True

    return plot


def _plot_trj_marker(wks, plot, x, y, angle, res):
    ''' Plots a single arrow head onto the a trajectory plot

    pgon = _plot_trj_marker(wks, plot, x, y, angle, res)

    wks :              The identifier returned from calling Ngl.open_wks.

    x,y :              The X and Y coordinates of the arrow heads centre.

    xMarker, yMarker : X and Y coordinates in data coordinates of the marker
polygon

    angle:             angle relative to the x axis of the marker

    res:               Resource list using following special resources:

    Special resources:

    trjArrowDirection : can be 'pos' or 'neg' to select direction of the arrows

    trjArrowXShapeF   : Array containing the x coordinates of the arrow head
relative to the heads centre.

    trjArrowYShapeF   : Array containing the y  coordinates of the arrow head
relative to the heads centre.
    '''

    # get plot centre in data coordinates
    xMid, yMid = [(ngl.get_float(plot, "tr" + ax + "MaxF")
                   + ngl.get_float(plot, "tr" + ax + "MinF")) / 2.
                          for ax in "XY"]

    # rotate arrow
    if res.trjArrowDirection == "neg":
        angle = angle - np.pi

    xMarker = copy.deepcopy(res.trjArrowXShapeF)
    yMarker = copy.deepcopy(res.trjArrowYShapeF)

    # scale marker
    xMarker = xMarker * res.trjArrowXScaleF * res.trjArrowSizeF + xMid
    yMarker = yMarker * res.trjArrowYScaleF * res.trjArrowSizeF + yMid

    xMarker_ndc, yMarker_ndc = ngl.datatondc(plot, xMarker, yMarker)

    # move centre of mass to origin
    xMarker_ndc, yMarker_ndc = [xMarker_ndc - np.mean(xMarker_ndc),
                                yMarker_ndc - np.mean(yMarker_ndc)]
    # rotate marker
    xMarker_ndc, yMarker_ndc = [
                    np.cos(angle) * xMarker_ndc - np.sin(angle) * yMarker_ndc,
                    np.sin(angle) * xMarker_ndc + np.cos(angle) * yMarker_ndc]

    # shift to final position
    xOffset_ndc, yOffset_ndc = ngl.datatondc(plot, x, y)
    xMarker_ndc += xOffset_ndc
    yMarker_ndc += yOffset_ndc

    # convert back to coordinates
    xMarker, yMarker = ngl.ndctodata(plot, xMarker_ndc, yMarker_ndc)

    # filter attributes from res
    for attr in dir(res):
        if not attr[0:2] in ["gs", "__"] or attr[0:3] == "gsn":
            delattr(res, attr)

    return ngl.add_polygon(wks, plot, xMarker, yMarker, res)


def _set_values(plot, res_dict):
    res = ngl.Resources()
    res.__dict__.update(res_dict)
    ngl.set_values(plot, res)
