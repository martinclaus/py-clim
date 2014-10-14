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

    cmap : Numpy Nx3 array with RGB values of colormap

    ncolor : Number of colors to produce. Note that the colormap actually have
             ncolor + 2 colors (foreground and background color added).

    saturation : saturation of the colormap in HSV space

    value : brightness of the colormap.

    bg_color : background color in RGB

    fg_color : foreground color in RGB
    '''
    hue = np.linspace(0., 360., ncolor, endpoint=False)
    cmap = np.zeros((ncolor + 2, 3))
    cmap[0, :] = bg_color
    cmap[1, :] = fg_color
    for i, h in enumerate(hue):
        cmap[i + 2, :] = ngl.hsvrgb(h, saturation, value)
    return cmap


def add_panel_label(wks, plot, panLabel="", factor=.05,
                    placement="ul", axfunc=min, rlist=None):
    ''' Adds a label to a plot similar to the result from the
    nglPanelFigureString resource. The box around the perimenter of the label
    is drawn by the text_ndc function and controlled by txPerim* attributes.

    add_panel_label(wks, plot, panLabel="", factor=(1., 1.),
                    placement="ul", axfunc=min, rlist=None)

    wks : workstation Id as returned by open_wks()

    plot : plot Id of the panel to add the label

    panLabel : String containing the panel label

    factor : ratio of margin to reference axis length

    placement : One of "ul", "ur", "ll" or "lr" to specify the corner where the
    label should be placed

    axfunct : function which is used to give the reference axis length based on
    the viewport width and height. This function should accept one list or
    tupel as argument.

    rlist : Resouce object that can be used to customize the label and its
    perimeter box. It can contain any resource that is accepted by text_ndc as
    a valid resource.
    '''
    vpXF, vpYF, vpWidthF, vpHeightF = [ngl.get_float(plot, name) for name in
                                       ["vpXF", "vpYF",
                                       "vpWidthF", "vpHeightF"]]

    margin = factor * axfunc([vpWidthF, vpHeightF])

    ul_edge = [vpXF + .5 * vpWidthF, vpYF + .5 * vpHeightF]
    if placement[0] == "u":
        ul_edge[1] = vpYF - margin
        tx_just = "Top"
    elif placement[0] == "l":
        ul_edge[1] = vpYF - vpHeightF + margin
        tx_just = "Bottom"
    if placement[1] == "l":
        ul_edge[0] = vpXF + margin
        tx_just = tx_just + "Left"
    elif placement[1] == "r":
        ul_edge[0] = vpXF + vpWidthF - margin
        tx_just = tx_just + "Right"
    if len(ul_edge) != 2:
        raise ValueError("placement is not in ['ul', 'ur', 'll', 'lr']")

    tx_res = {"txFontHeightF": .02,
              "txPerimOn": True,
              "txBackgroundFillColor": 0,
              "txJust": tx_just}
    if rlist:
        tx_res.update(rlist.__dict__)

    ngl.text_ndc(wks, panLabel, *ul_edge, rlistc=_dict2Resource(tx_res))


def add_axis(wks, plot, ax="XB", offset=0., res=None):
    val_ax = ("XT", "XB", "YL", "YR")
    if not res:
        res = {}
    else:
        res = _resource2dict(res)
    resp = {}
    keys = ["vpXF", "vpYF", "vpWidthF", "vpHeightF", "trXMinF", "trXMaxF",
            "trYMinF", "trYMaxF"]
    for a in val_ax:
        for k in ("LabelFontHeight", "MajorOutwardLength",
                  "MinorOutwardLength"):
            keys.append("tm{}{}F".format(a, k))
    for k in keys:
        resp[k] = ngl.get_float(plot, k)
    for k in ("tm" + a + k for a in val_ax for k in ("Values", "MinorValues")):
        resp[k] = ngl.get_float_array(plot, k)
    for k in ("tm{}MinorPerMajor".format(a) for a in val_ax):
        resp[k] = ngl.get_integer(plot, k)
    for k in ("tm{}MinorOn".format(a) for a in val_ax):
        resp[k] = (ngl.get_integer(plot, k) == 1)
    for k in ("tm" + a + "Labels".format(a) for a in val_ax):
        resp[k] = ngl.get_string_array(plot, k)
    resp.update(res)

    for a in ("XT", "XB", "YL", "YR"):
        resp["tm{}Mode".format(a)] = "Explicit"
        resp["tm{}On".format(a)] = (a == ax)
        resp["tm{}BorderOn".format(a)] = (a == ax)

    resp["nglDraw"] = False
    resp["nglFrame"] = False

    blank_plot = ngl.blank_plot(wks, _dict2Resource(resp))
    amres = {"amJust": "CenterCenter"}
    if ax[1].lower() in "lt":
        ampos_sig = -1.
    else:
        ampos_sig = 1.
    if ax[0].lower() == "x":
        amres["amOrthogonalPosF"] = ampos_sig * offset
    else:
        amres["amParallelPosF"] = ampos_sig * offset

    return ngl.add_annotation(plot, blank_plot, _dict2Resource(amres))


def set_plot_position(plot, vpXY=None, vpWH=None):
    '''Change the viewport location and width. If one is omitted, the value
    will be retained.

    set_plot_position(plot, vpXY=None, vpWH=None)

    plot : plot Id of the plot to modify

    vpXY : tupel of the coordinates of the upper left corner of the viewport in
    NDC space

    vpWH : tupel of viewports width and height in NDC space
    '''
    attrib = ["vpXF", "vpYF", "vpWidthF", "vpHeightF"]
    rlist = dict(zip(attrib, [ngl.get_float(plot, name) for name in attrib]))

    if vpXY:
        rlist["vpXF"], rlist["vpYF"] = vpXY
    if vpWH:
        rlist["vpWidthF"], rlist["vpHeightF"] = vpWH
    _set_values(plot, rlist)


def cn_sym_min_max(x, res, ncontours=15, outside=True, aboutZero=True):
    '''Adjust the contour level configuration of a plot resource to give nice
    contour levels symmetric about zero. The contour levels are computed with
    NGL.nice_cntr_levels().

    cn_sym_min_max(x, res, ncontours=15, outside=True, aboutZero=True)

    x : data used in the contour plot.

    res : Resource object that will be updated accordingly.

    outside : Whether the maxima is outside or inside the contour range. Will
    be passed to NGL.nice_cntr_levels().

    aboutZero : Whether the contour levels will be centered about Zero. Will be
    passed to NGL.nice_cntr_levels().
    '''
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


def cn_neg_dash_contours(plot, pattern=2):
    '''Changes the line dash pattern of the negative contour lines.

    cn_neg_dash_contours(plot, pattern=2)

    plot : plot Id of the plot to modify

    pattern : Pattern index. See http://www.ncl.ucar.edu/Document/Graphics/Images/dashpatterns.png
    for a list of available patterns.
    '''
    levels = ngl.get_float_array(plot, "cnLevels")

    patterns = np.zeros((len(levels)), 'i')
    patterns[:] = 0     # solid line
    for i in xrange(len(levels)):
        if (levels[i] < 0.):
            patterns[i] = pattern
    rlist = {"cnLineDashPatterns": patterns,
             "cnMonoLineDashPattern": False}
    _set_values(plot, rlist)


def cn_thick_zero_contour(plot, factor=2.):
    ''' Increases the thickness of the zero contour line.

    cn_thick_zero_contour(plot, factor=2.)

    plot : plot Id of the plot to modify

    factor : float that will be multiplied to the present thickness
    '''
    cn_line_thickness(plot, val=0., factor=factor)


def cn_line_thickness(plot, val=0., factor=2.):
    ''' Increases the thickness of contour lines corresponding to a list of
    values

    cn_line_thickness(contour, val=0., factor=2.)

    plot : plot Id of the plot to modify

    val : scalar or list of values, which contour lines should be modified.

    factor : scaling factor for line thickness
    '''
    cnLevels = ngl.get_float_array(plot, "cnLevels")
    cnLineThicknesses = ngl.get_float_array(plot, "cnLineThicknesses")

    try:
        (v for v in val)
    except TypeError:
        val = [val]

    for i, thickness in enumerate(cnLineThicknesses[:]):
        if cnLevels[i] in val:
            cnLineThicknesses[i] = thickness * factor

    rlist = {"cnMonoLineThickness": False,
             "cnLineThicknesses": cnLineThicknesses}
    _set_values(plot, rlist)


def tm_lat_lon_labels(plot, ax="XB", direction=None, fmt="%d"):
    '''Add degree symbols to the tickmark labels of one axis of a plot. This
    function has to be called for each axis, that should get degree tickmark
    labels

    tm_lat_lon_labels(plot, ax="XB", direction="zonal")

    plot : plot Id of the plot to modify

    ax : one of "XB", "XT", "YL", "YR". Specifies the axis to work on.

    direction : String that starts with either "z" or "m", to indicate the
    direction of the axis (zonal or meridional). If omitted, a x-axis will be
    assumed to be zonal. If the string starts with any other character, there
    will be no hemisperic lables added, i.e N, S, W or E.

    fmt : format string of the number
    '''
    def do_nothing(x):
        return x
    func_code = ngl.get_string(plot, "tm" + ax + "LabelFuncCode")
    fmt_s = func_code.join([fmt, "S", "o", "N", ""])
    if not direction:
        if ax[0].lower() == "x":
            direction = "zonal"
        else:
            direction = "meridional"
    if direction[0].lower() == "z":
        hem_label = ("W", "E")
        func = abs
    elif direction[0].lower() == "m":
        hem_label = ("S", "N")
        func = abs
    else:
        hem_label = ("", "")
        func = do_nothing

    tm_values = ngl.get_float_array(plot, "tm" + ax + "Values")
    mtm_values = ngl.get_float_array(plot, "tm" + ax + "MinorValues")

    labels = [fmt_s % func(val) + hem_label[0] if val < 0.
              else fmt_s % func(val) + hem_label[1] for val in tm_values]

    rlist = {"tm" + ax + "Mode": "Explicit",
             "tm" + ax + "Values": tm_values,
             "tm" + ax + "MinorValues": mtm_values,
             "tm" + ax + "Labels": labels}
    _set_values(plot, rlist)


def tm_deg_labels(plot, ax="YL", fmt="%3.1f"):
    '''Add degree symbols an format the tickmark labels of an given axis

    tm_deg_labels(plot, ax="YL", fmt="%3.1f"):

    plot : plot Id of the plot to modify

    ax : one of "XB", "XT", "YL", "YR". Specifies the axis to work on.

    fmt : format string of the number
    '''
    tm_lat_lon_labels(plot, ax, direction="any", fmt=fmt)


def lg_create_custom_legend(wks, vpXY=(.1, .9), vpWH=(.8, .8), shape=(3, 2),
                     labels=["a", "b", "c", "d", "e"], rlistc=None):
    if len(labels) > shape[0] * shape[1]:
        raise ValueError("To many legend items for given shape!")
    # set default values
    if not rlistc:
        rlist = {}
    else:
        rlist = _resource2dict(rlistc)
    rlist["lgLabelStrings"] = labels
    rlist["vpXY"] = vpXY
    rlist["vpWH"] = vpWH
    _set_custom_legend_defaults(rlist)

    # compute locations
    if rlist["lgOrientation"][0].lower() == "v":
        rlist["min_ax"] = vpWH[0]
    else:
        rlist["min_ax"] = vpWH[1]
    rlist["margin"] = {}
    for key in ("Top", "Bottom", "Left",  "Right"):
        lgKey = "lg{}MarginF".format(key)
        rlist["margin"][key] = rlist[lgKey] * rlist["min_ax"]

    # calculate positions
    itemPanelXY = [vpXY[0] + rlist["margin"]["Left"],
                   vpXY[1] - rlist["margin"]["Top"]]
    itemPanelWH = [vpWH[0]
                    - rlist["margin"]["Left"] - rlist["margin"]["Right"],
                   vpWH[1]
                    - rlist["margin"]["Top"] - rlist["margin"]["Bottom"]]

    _lg_add_legend_perim(wks, vpXY, vpWH, rlist)
    _lg_add_legend_title(wks, (itemPanelXY, itemPanelWH), rlist)

    itemsWH = (itemPanelWH[0] / shape[1], itemPanelWH[1] / shape[0])
    itemsXY = ([itemPanelXY[0] + i * itemsWH[0] for i in range(shape[1])],
               [itemPanelXY[1] - i * itemsWH[1] for i in range(shape[0])])

    convert_position = {"Left": "Bottom", "Bottom": "Left",
                        "Right": "Top", "Top": "Right", "Center": "Center"}
    if rlist["lgOrientation"][0].lower() == "v":
        rlist["valid_positions"] = ("Left", "Right", "Center")
    else:
        rlist["valid_positions"] = ("Top", "Bottom", "Center")
    if rlist["lgLabelPosition"] not in rlist["valid_positions"]:
        rlist["lgLabelPosition"] = convert_position[rlist["lgLabelPosition"]]

    for i in range(shape[0]):
        for j in range(shape[1]):
            if rlist["lgOrientation"][0].lower() == "h":
                iItem = j + i * shape[1]
            else:
                iItem = i + j * shape[0]
            if (iItem + 1) > rlist["lgItemCount"]:
                break
            _lg_add_legend_item(wks, rlist["lgItemOrder"][iItem],
                                (itemsXY[0][j], itemsXY[1][i]), itemsWH, rlist)


def _lg_add_legend_perim(wks, (x, y), (w, h), res):
    bres = {}
    bres["gsEdgesOn"] = res["lgPerimOn"]
    for key in _filter_keys(res, "lgPerimFill"):
        bres["gsFill" + key] = res["lgPerimFill" + key]
    for key in _filter_keys(res, "lgPerim", exceptions=("On"),
                            falsepositive="Fill"):
        bres["gsEdge" + key] = res["lgPerim" + key]
    ngl.polygon_ndc(wks, *_xywh2corners((x, y), (w, h)),
                    rlistc=_dict2Resource(bres))


def _lg_add_legend_title(wks, (iXY, iWH), res):
    if not res["lgTitleOn"]:
        return
    x, y = list(res["vpXY"])
    w, h = list(res["vpWH"])
    offset = {"Top": (0, -res["lgTitleOffsetF"] * res["vpWH"][1]),
              "Bottom": (0, res["lgTitleOffsetF"] * res["vpWH"][1]),
              "Left": (res["lgTitleOffsetF"] * res["vpWH"][0], 0),
              "Right": (-res["lgTitleOffsetF"] * res["vpWH"][0], 0)}
    if res["lgTitlePosition"] in ("Top", "Bottom"):
        w = res["lgTitleExtentF"] * (res["vpWH"][1]
                        - res["margin"]["Top"] - res["margin"]["Bottom"])
        if res["lgTitlePosition"] == "Top":
            iXY[1] = iXY[1] - h + offset["Top"][1]
            iWH[1] = iWH[1] - h + offset[res["lgTitlePosition"]][1]
        elif res["lgTitlePosition"] == "Bottom":
            y = res["vpXY"][1] - res["vpWH"][1] + h
            iWH[1] = iWH[1] - h - offset[res["lgTitlePosition"]][1]
    elif res["lgTitlePosition"] in ("Left", "Right"):
        w = res["lgTitleExtentF"] * (res["vpWH"][0]
                        - res["margin"]["Left"] - res["margin"]["Right"])
        if res["lgTitlePosition"] == "Left":
            iXY[0] = iXY[0] + w + offset["Left"][0]
            iWH[0] = iWH[0] - w - offset[res["lgTitlePosition"]][0]
        elif res["lgTitlePosition"] == "Right":
            x = res["vpXY"][0] + res["vpWH"][0] - w
            iWH[0] = iWH[0] - w + offset[res["lgTitlePosition"]][0]
    else:
        x, y, w, h = 0., 0., 0., 0.
    tires = {}
    # copy legend title properties
    for key in _filter_keys(res, "lgTitle", exceptions=("ExtentF", "On",
                                                        "Position", "String",
                                                        "OffsetF")):
        tires["tx" + key] = res["lgTitle" + key]
    ngl.text_ndc(wks, res["lgTitleString"], x + w / 2., y - h / 2.,
                 rlistc=_dict2Resource(tires))


def _lg_add_legend_item(wks, iItem, (x, y), (w, h), res):
    bx, by, bw, bh = x, y, w, h
    if res["lgOrientation"][0].lower() == "v":
        by = by - (1 - res["lgBoxMajorExtentF"]) * bh / 2.
        bh = res["lgBoxMajorExtentF"] * bh
        bw = res["lgBoxMinorExtentF"] * bw
        if res["lgLabelPosition"] == "Left":
            bx = bx + w - bw
    else:
        bx = bx + (1 - res["lgBoxMajorExtentF"]) * bw / 2.
        bw = res["lgBoxMajorExtentF"] * bw
        bh = res["lgBoxMinorExtentF"] * bh
        if res["lgLabelPosition"] == "Top":
            by = by - h + bh

    if res["lgBoxLinesOn"]:
        bres = {"gsEdgesOn": True}
        for key in _filter_keys(res, "lgBox", exceptions=("LinesOn",
                                                          "MajorExtentF",
                                                          "MinorExtendF")):
            if key == "Background":
                bres["gsFillColor"] = res["lgBox" + key]
            elif key[:4] == "Line":
                bres["gsEdge" + key[:4]] = res["lgBox" + key]
        ngl.polygon_ndc(wks, *_xywh2corners((bx, by), (bw, bh)),
                        rlistc=_dict2Resource(bres))

    _lg_add_legend_item_line(wks, iItem, (bx, by), (bw, bh), res)
    _lg_add_legend_item_label(wks, iItem, (bx, by), (bw, bh), res)


def _lg_add_legend_item_line(wks, iItem, (x, y), (w, h), res):
    lbx, lby, lbw, lbh = x, y, w, h
    if res["lgOrientation"][0].lower() == "v":
        lx = (lbx, lbx + lbw)
        ly = (lby - lbh / 2., lby - lbh / 2.)
    else:
        lx = (lbx + lbw / 2., lbx + lbw / 2.)
        ly = (lby, lby - lbh)
    linesOn = res["lgItemTypes"][iItem][-5:].lower() == "lines"
    markerOn = res["lgItemTypes"][iItem][:4].lower() == "mark"
    lineLabelsOn = res["lgLineLabelsOn"]

    if linesOn:
        lres = {}
        for key in _filter_keys(res, "lgLine", falsepositive="Label"):
            try:
                lres[_plural2singular("gsLine" + key)] = res["lgLine" + key][iItem]
            except TypeError:
                pass
        lres["gsLineDashPattern"] = res["lgDashIndexes"][iItem]
        ngl.polyline_ndc(wks, lx, ly, rlistc=_dict2Resource(lres))

    if markerOn:
        lres = {}
        for key in _filter_keys(res, "lgMarker"):
            try:
                lres[_plural2singular("gsMarker" + key)] = res["lgMarker" + key][iItem]
            except:
                pass
        if lineLabelsOn:
            mx, my = [[l[0] + fac * (l[1] - l[0]) for fac in [.2, .8]]
                        for l in [lx, ly]]
        else:
            mx, my = [l[0] + .5 * (l[1] - l[0]) for l in [lx, ly]]
        ngl.polymarker_ndc(wks, mx, my, rlistc=_dict2Resource(lres))

    if lineLabelsOn:
        txres = {}
        for key in _filter_keys(res, "lgLineLabel", exceptions=("sOn",
                                                                "Strings")):
            try:
                txres[_plural2singular("tx" + key)] = res["lgLineLabel" + key][iItem]
            except TypeError:
                txres["tx" + key] = res["lgLineLabel" + key]
        txres["txBackgroundFillColor"] = max(res["lgBoxBackground"], 0)
        if res["lgOrientation"][0].lower() == "h":
            _set_default(txres, "txAngleF", 90.)
        llx, lly = [(l[0] + l[1]) / 2. for l in [lx, ly]]
        ngl.text_ndc(wks, res["lgLineLabelStrings"][iItem], llx, lly,
                     rlistc=_dict2Resource(txres))


def _lg_add_legend_item_label(wks, iItem, (x, y), (w, h), res):
    if (not res["lgLabelsOn"]) or (not iItem % res["lgLabelStride"] == 0):
        return
    txres = {}
    for key in _filter_keys(res, "lgLabel",
                            exceptions=("sOn", "Strings","Alignment",
                                        "AutoStride", "OffsetF", "Position",
                                        "Stride")):
        try:
            txres[_plural2singular("tx" + key)] = res["lgLineLabel" + key][iItem]
        except TypeError:
            txres["tx" + key] = res["lgLineLabel" + key]

    txJust, txy = _lg_get_item_label_justification_position(res, (x, y),
                                                            (w, h))
    _set_default(txres, "txJust", txJust)
    ngl.text_ndc(wks, res["lgLabelStrings"][iItem], *txy,
                 rlistc=_dict2Resource(txres))


def _lg_get_item_label_justification_position(res, (x, y), (w, h)):
    invert_position = {"Left": "Right", "Bottom": "Top",
                        "Right": "Left", "Top": "Bottom", "Center": "Center"}
    if res["lgOrientation"][0].lower() == "v":
        conv_align = {"ItemCenters": "Center", "AboveItems": "Bottom",
                      "BelowItems": "Top"}
        pos2just = dict((k, conv_align[res["lgLabelAlignment"]]
                            + invert_position[k])
                        for k in res["valid_positions"])
    else:
        conv_align = {"ItemCenters": "Center", "AboveItems": "Right",
                      "BelowItems": "Left"}
        pos2just = dict((k, invert_position[k]
                            + conv_align[res["lgLabelAlignment"]])
                        for k in res["valid_positions"])

    if invert_position[res["lgLabelPosition"]] in ("Right", "Top"):
        offset = -res["min_ax"] * res["lgLabelOffsetF"]
    else:
        offset = res["min_ax"] * res["lgLabelOffsetF"]
    xy_coord = {"Left": (x + offset, y - h / 2),
                "Right": (x + w + offset, y - h / 2.),
                "Center": (x + w / 2., y - h / 2. + offset),
                "Top": (x + w / 2., y + offset),
                "Bottom": (x + w / 2., y - h + offset)}
    return pos2just[res["lgLabelPosition"]], xy_coord[res["lgLabelPosition"]]


def _set_custom_legend_defaults(rlist):
    ngl._set_legend_res(rlist, rlist)
    defaults = {
                "lgBoxBackground": -1,
#                "lgBoxLineColor": 1,
#                "lgBoxLineDashPattern": 0,
#                "lgBoxLineDashSegLenF": 0.15,
#                "lgBoxLineThicknessF": 1.0,
                "lgBoxLinesOn": False,
                "lgBoxMajorExtentF": .5,
                "lgBoxMinorExtentF": .6,
#                "lgDashIndexes": [0 for l in rlist["lgLabelStrings"]],
                "lgItemCount": len(rlist["lgLabelStrings"]),
                "lgItemOrder": range(len(rlist["lgLabelStrings"])),
#                "lgItemTypes": ["Lines" for l in rlist["lgLabelStrings"]],
                "lgLabelAlignment": "ItemCenters",
#                "lgLabelAngleF": 0.0,
#                "lgLabelConstantSpacingF": 0.0,
#                "lgLabelDirection": "Across",
#                "lgLabelFont": "pwritx",
#                "lgLabelFontAspectF": 1.0,
#                "lgLabelFontColor": 1,
#                "lgLabelFontHeightF": 0.02,
#                "lgLabelFontQuality": "High",
#                "lgLabelFontThicknessF": 1.0,
#                "lgLabelFuncCode": ":",
#                "lgLabelJust": "CentreCentre",
                "lgLabelOffsetF": 0.02,
                "lgLabelPosition": "Right",
                "lgLabelStride": 1,
                "lgLabelsOn": True,
                "lgLegendOn": True,
                "lgLineColors": range(2, len(rlist["lgLabelStrings"]) + 2),
#                "lgLineDashSegLenF": 0.15,
#                "lgLineLabelConstantSpacingF": 0.0,
#                "lgLineLabelFont": "pwritx",
#                "lgLineLabelFontAspectF": 1.0,
                "lgLineLabelFontHeights": [0.01
                                           for l in rlist["lgLabelStrings"]],
#                "lgLineLabelFontQuality": "High",
#                "lgLineLabelFontThicknessF": 1.0,
#                "lgLineLabelFuncCode": ":",
                "lgLineLabelStrings": rlist["lgLabelStrings"],
                "lgLineLabelsOn": False,
#                "lgMarkerIndexes": [1 for l in rlist["lgLabelStrings"]],
                "lgMarkerColors": range(2, len(rlist["lgLabelStrings"]) + 2),
                "lgOrientation": "Vertical",
                "lgPerimOn": False,
#                "lgPerimColor": 1,
#                "lgPerimDashPattern": 0,
#                "lgPerimDashSegLenF": 0.15,
                "lgPerimFillColor": -1,
#                "lgPerimThicknessF": 1.0,
#                "lgTitleAngleF": 0.0,
#                "lgTitleConstantSpacingF": 0.0,
#                "lgTitleDirection": "Across",
                "lgTitleExtentF": 0.15,
#                "lgTitleFont": "pwritx",
#                "lgTitleFontAspectF": 1.0,
#                "lgTitleFontColor": 1,
#                "lgTitleFontHeightF": .025,
#                "lgTitleFontQuality": "High",
#                "lgTitleFontThicknessF": 1.0,
#                "lgTitleFuncCode": ":",
#                "lgTitleJust": "CenterCenter",
                "lgTitleOffsetF": 0.03,
                "lgTitlePosition": "Top",
                "lgTitleString": "",
                }
    for key, val in defaults.items():
        _set_default(rlist, key, val)
    _set_default(rlist, "lgTitleOn", "lgTitleString" in rlist
                    and rlist["lgTitleString"].strip())
    for key in ("Top", "Bottom", "Left",  "Right"):
        _set_default(rlist, "lg{}MarginF".format(key), .05)
    _set_mono_default(rlist, "lgMonoDashIndex", "lgDashIndex",
                      "lgDashIndexes", 0)
    _set_mono_default(rlist, "lgMonoItemType", "lgItemType",
                      "lgItemTypes", "Lines")
    _set_mono_default(rlist, "lgMonoLineColor", "lgLineColor",
                      "lgLineColors", 1)
    _set_mono_default(rlist, "lgMonoLineDashSegLen", "lgLineDashSegLenF",
                      "lgLineDashSegLens", 0.15)
    _set_mono_default(rlist, "lgMonoLineLabelFontColor",
                      "lgLineLabelFontColor", "lgLineLabelFontColors", 1)
    _set_mono_default(rlist, "lgMonoLineLabelFontHeight",
                      "lgLineLabelFontHeightF", "lgLineLabelFontHeights", 0.01)
    _set_mono_default(rlist, "lgMonoLineThickness", "lgLineThicknessF",
                      "lgLineThicknesses", 1.0)
    _set_mono_default(rlist, "lgMonoMarkerColor", "lgMarkerColor",
                      "lgMarkerColors", 1)
    _set_mono_default(rlist, "lgMonoMarkerIndex", "lgMarkerIndex",
                      "lgMarkerIndexes", 0)
    _set_mono_default(rlist, "lgMonoMarkerSize", "lgMarkerSizeF",
                      "lgMarkerSizes", 0.01)
    _set_mono_default(rlist, "lgMonoMarkerThickness", "lgMarkerThicknessF",
                      "lgMarkerThicknesses", 1.)
    _set_default(rlist, "lgLineLabelFontColors", rlist["lgLineColors"])


def lg_create_legend_nd(wks, labels, x, y, shape, rlist=None):
    # Set defaults
    if not rlist:
        rlist = {}
    else:
        rlist = _resource2dict(rlist)
    ngl._set_legend_res(rlist, rlist)
    _set_default(rlist, "lgOrientation", "Vertical")
    _set_default(rlist, "vpWidthF", 0.6)
    _set_default(rlist, "vpHeightF", 0.6)
    _set_default(rlist, "lgItemCount", len(labels))
    _set_default(rlist, "lgLineColors", [i + 2
                                         for i in range(rlist["lgItemCount"])])
    _set_default(rlist, "lgLineLabelFontColors", rlist["lgLineColors"])

    if rlist["lgOrientation"].lower() == "vertical":
        nItems, nLegends = shape
        lwidth = rlist["vpWidthF"] / nLegends
        lheight = rlist["vpHeightF"]
        xpos = [x + i * lwidth for i in range(nLegends)]
        ypos = [y for i in range(nLegends)]
    else:
        nLegends, nItems = shape
        lwidth = rlist["vpWidthF"]
        lheight = rlist["vpHeightF"] / nLegends
        ypos = [y - i * lheight for i in range(nLegends)]
        xpos = [x for i in range(nLegends)]

    istart = 0
    lg = []
    for xi, yi in zip(xpos, ypos):
        nItem = min(nItems, rlist["lgItemCount"] - istart)
        if nItem <= 0:
            continue
        iend = istart + nItem
        lres = rlist.copy()
        if rlist["lgOrientation"].lower() == "vertical":
            lres["vpWidthF"] = lwidth
            lres["vpHeightF"] = lheight * nItem / nItems
        else:
            lres["vpWidthF"] = lwidth * nItem / nItems
            lres["vpHeightF"] = lheight
        for key in ("lgDashIndexes", "lgItemPositions", "lgItemTypes",
                    "lgLabelStrings", "lgLineColors", "lgLineDashSegLens",
                    "lgLineLabelFontColors", "lgLineLabelFontHeights",
                    "lgLineLabelStrings", "lgLineThicknesses",
                    "lgMarkerColors", "lgMarkerIndexes", "lgMarkerSizes",
                    "lgMarkerThicknesses", ):
            if key in rlist:
                lres[key] = rlist[key][istart:iend]
        lres["lgItemCount"] = nItem

        lg.append(ngl.legend_ndc(wks, nItem, labels[istart:iend],
                                 xi, yi, _dict2Resource(lres)))
        istart = iend
    return lg


def lb_create_labelbar(wks, vpXY, vpWH, nboxes=11, levels=(-1., 1.),
                       fmt_str="{}", rlist=None):
    '''Creates a label bar.

    lb = lb_create_labelbar(wks, vpXY, vpWH, nboxes=11, levels=(-1., 1.),
                       frm_str="{}", rlist=None)

    wks : workstation Id as returned by open_wks()

    vpXY : tupel of upper left coordinates of the label bar's view port in NDC

    vpWH : tupel containing width and height of the label bar's view port
    in NDC

    nboxes : number of boxes

    levels : tupel containing the data extrems

    fmt_str : format string used to format the numbers

    rlist : Resource object containing additional resources that are accepted
    by labelbar_ndc()
    '''
    # add label bar for depth
    levels = np.linspace(*levels, num=nboxes)
    labels = [fmt_str.format(val) for val in levels]
    labres = {"lbAutoManage": False,
              "vpWidthF": vpWH[0],
              "vpHeightF": vpWH[1],
              "lbMonoFillPattern": 21,
              "lbOrientation": "vertical",
              "lbFillColors": map_data_to_color(wks, levels),
              "lbLabelStride": 10}
    if rlist:
        labres.update(rlist.__dict__)
    lb = ngl.labelbar_ndc(wks, nboxes, labels, *vpXY,
                          rlistc=_dict2Resource(labres))
    return lb


def lb_set_phase_labels(plot):
    '''Changes the labels of a color bar to -pi, -pi/2, 0, pi/2 and pi. The
    label bar must already have only these lables.

    lb_set_phase_labels(plot)

    plot : plot id of object that controlls the lable bar
    '''
    lstride = ngl.get_integer(plot, "lbLabelStride")
    label = ngl.get_string_array(plot, "lbLabelStrings")
    fun_code = ngl.get_string(plot, "lbLabelFuncCode")
    pi_str = fun_code.join(["", "F33", "p"])
    pi_half_str = fun_code.join(["", "H2V15F33", "p", "H-15V-1", "_",
                                 "H-16V-30", "2"])
    label[::lstride] = ["-" + pi_str, "-" + pi_half_str, "0",
                        pi_half_str, pi_str]
    rlist = {"cnExplicitLabelBarLabelsOn": True,
             "lbLabelStrings": label}
    _set_values(plot, rlist)


def lb_format_labels(plot, fmt, minmax=None):
    '''Applies a formatting to the lables of a label bar

    lb_format_labels(plot, fmt, minmax=None)

    plot : plot id of object that controlls the lable bar

    fmt : formatting string

    minmax : values of the maximum and minimum. Required, if cnLabelBarEndStyle
    is set to IncludeMinMaxLabels. If not provided in this case, a ValueError
    will be raised.
    '''
    levels = [float(s) for s in ngl.get_string_array(plot, "cnLevels")]
    labels = [fmt % lev for lev in levels]
    if ngl.get_string(plot, "cnLabelBarEndStyle") == "IncludeMinMaxLabels":
        if not minmax:
            raise ValueError("You need to provide minmax,"
                           + " since cnLabelBarEndStyle is set to "
                           + "IncludeMinMaxLabels!")
        minlabel = [fmt % minmax[0]]
        minlabel.extend(labels)
        labels = minlabel
        labels.append(fmt % minmax[1])
    rlist = {"cnExplicitLabelBarLabelsOn": True,
             "lbLabelStrings": labels}
    _set_values(plot, rlist)


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
    ngl.set_values(plot, _dict2Resource(res_dict))


def _dict2Resource(dic):
    res = ngl.Resources()
    res.__dict__.update(dic)
    return res


_resource2dict = ngl._crt_dict


def _set_default(rdic, attrib, val):
    if not attrib in rdic:
        rdic[attrib] = val


def _set_mono_default(rdic, mono_attrib, single_attrib, arr_attrib, val):
    if not (single_attrib in rdic or mono_attrib in rdic or arr_attrib in rdic):
        return
    _set_default(rdic, single_attrib, val)
    if arr_attrib not in rdic or (mono_attrib in rdic and rdic[mono_attrib]):
        rdic[arr_attrib] = [rdic[single_attrib]
                            for i in xrange(rdic["lgItemCount"])]


def _plural2singular(word):
    out = ""
    dic = {"Colors": "Color",
           "Lens": "LenF",
           "Thicknesses": "ThicknessF",
           "Heights": "HeightF",
           "Indexes": "Index",
           "Sizes": "SizeF",
           "Strings": "String"}
    istart = -1
    for key, value in dic.items():
        istart = word.rfind(key)
        if not istart == -1:
            out = word[:istart] + dic[key]
            break
    if not out:
        raise ValueError("No plural defined for word '{}'".format(word))
    else:
        return out


def _xywh2corners((x, y), (w, h)):
    return ((x, x + w, x + w, x, x), (y, y, y - h, y - h, y))


def _filter_keys(dic, pattern, exceptions=(), falsepositive=""):
    plen = len(pattern)
    for key in (k[plen:] for k in dic.keys() if len(k) > plen
                                             and k[:plen] == pattern):
        if (falsepositive and len(key) >= len(falsepositive) and
            key[:len(falsepositive)] == falsepositive):
            continue
        if exceptions and key in exceptions:
            continue
        yield key
