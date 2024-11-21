import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..l1b import *
from ..gis import *
from ..misc import *

__all__ = ["dem_transect"]

def coherence(waveform, *,
              ax: plt.Axes = None,
              plot_properties: dict[str, dict[str, any]] = \
                    dict(swath=dict(color="tab:blue",
                                    marker='.',
                                    markersize=5,
                                    linewidth=1),
                         poca=dict(color="tab:green",
                                   marker='o',
                                   markersize=5,
                                   linewidth=1),
                         excluded=dict(color="tab:pink",
                                       marker='x',
                                       markersize=5,
                                    #    linestyle="solid",
                                       linewidth=.6),
                         omitted=dict(hatch='///',
                                      linewidth=.4,
                                      edgecolor="xkcd:light grey",
                                      facecolor=None),
                         threshold=dict(hatch='\\\\\\',
                                        linewidth=.4,
                                        edgecolor="xkcd:light grey",
                                        facecolor=None))
            ):
    if ax is None:
        ax = plt.subplots()[1]
    h_smooth = gauss_filter_DataArray(waveform.coherence_waveform_20_ku, "ns_20_ku", 35, 35).plot(c=".7", label="smooth")[0]
    h_list = [h_smooth]
    try:
        if plot_properties["threshold"]["facecolor"] is None:
            plot_properties["threshold"]["facecolor"] = ax.get_facecolor()
        y0 = -1
        h_thr = ax.add_patch(mpl.patches.Rectangle((-100, y0),
                                                     waveform.ns_20_ku[-1]+200, waveform.coherence_threshold-y0,
                                                     **plot_properties["threshold"], label="threshold"))
        h_list.append(h_thr)
    except KeyError:
        pass
    try:
        if waveform.swath_start[0] > 0:
            if plot_properties["omitted"]["facecolor"] is None:
                plot_properties["omitted"]["facecolor"] = ax.get_facecolor()
            x0 = -100
            h_omitted = ax.add_patch(mpl.patches.Rectangle((x0, -1),
                                                        waveform.swath_start[0]-x0, 3,
                                                        **plot_properties["omitted"], label="omitted"))
            h_list.insert(1, h_omitted)
    except AttributeError:
        pass
    try:
        excluded = waveform.where(waveform.exclude_mask)
        h_excl, = ax.plot(excluded.ns_20_ku, excluded.coherence_waveform_20_ku[0], ls='',
                          **plot_properties["excluded"], label="excluded")
        h_list.insert(0, h_excl)
        swath = waveform.where(~waveform.exclude_mask)
        h_swath, = ax.plot(swath.ns_20_ku, swath.coherence_waveform_20_ku[0], ls='', **plot_properties["swath"],
                           label="swath")
        h_list.insert(0, h_swath)
    except KeyError:
        h_all, = ax.plot(waveform.ns_20_ku, waveform.coherence_waveform_20_ku[0], ls='', **plot_properties["swath"],
                         label="all")
        h_list.insert(0, h_all)
    try:
        poca = waveform.sel(ns_20_ku=waveform.poca_idx)
        h_poca, = ax.plot(poca.ns_20_ku, poca.coherence_waveform_20_ku[0], ls='', **plot_properties["poca"],
                          label="POCA")
        h_list.insert(0, h_poca)
    except KeyError:
        pass
    ax.set_xlim([waveform.ns_20_ku[0], waveform.ns_20_ku[-1]])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("sample number (ns_20_ku)")
    ax.set_ylabel("coherence")
    ax.legend(handles=h_list)
    ax.set_title(f"id: {waveform.time_20_ku.values[0]}")
    return ax


def power(waveform, *,
              ax: plt.Axes = None,
              logarithmic: bool = True,
              plot_properties: dict[str, dict[str, any]] = \
                    dict(swath=dict(color="tab:blue",
                                    marker='.',
                                    markersize=5,
                                    linewidth=1),
                         poca=dict(color="tab:green",
                                   marker='o',
                                   markersize=5,
                                   linewidth=1),
                         excluded=dict(color="tab:pink",
                                       marker='x',
                                       markersize=5,
                                    #    linestyle="solid",
                                       linewidth=.6),
                         omitted=dict(hatch='///',
                                      linewidth=.4,
                                      edgecolor="xkcd:light grey",
                                      facecolor=None),
                         threshold=dict(hatch='\\\\\\',
                                        linewidth=.4,
                                        edgecolor="xkcd:light grey",
                                        facecolor=None))
            ):
    if ax is None:
        ax = plt.subplots()[1]
    if logarithmic:
        waveform["P"] = 10*np.log10(waveform.power_waveform_20_ku)
    else:
        waveform["P"] = waveform.power_waveform_20_ku
    h_smooth = gauss_filter_DataArray(waveform.P, "ns_20_ku", 35, 35).plot(c=".7", label="smooth")[0]
    h_list = [h_smooth]
    try:
        if plot_properties["threshold"]["facecolor"] is None:
            plot_properties["threshold"]["facecolor"] = ax.get_facecolor()
        assert(waveform.power_threshold[0] == "snr")
        if logarithmic:
            power_threshold = waveform.noise_power_20_ku[0]+10*np.log10(waveform.power_threshold[1])
            y0 = -250
        else:
            power_threshold = waveform.noise_power_20_ku[0]*waveform.power_threshold[1]
            y0 = -1
        h_thr = ax.add_patch(mpl.patches.Rectangle((-100, y0),
                                                     waveform.ns_20_ku[-1]+200, power_threshold-y0,
                                                     **plot_properties["threshold"], label="threshold"))
        h_list.append(h_thr)
    except KeyError:
        pass
    try:
        if waveform.swath_start[0] > 0:
            if plot_properties["omitted"]["facecolor"] is None:
                plot_properties["omitted"]["facecolor"] = ax.get_facecolor()
            x0 = -100
            h_omitted = ax.add_patch(mpl.patches.Rectangle((x0, -250),
                                                        waveform.swath_start[0]-x0, 252,
                                                        **plot_properties["omitted"], label="omitted"))
            h_list.insert(1, h_omitted)
    except AttributeError:
        pass
    try:
        excluded = waveform.where(waveform.exclude_mask)
        h_excl, = ax.plot(excluded.ns_20_ku, excluded.P[0], ls='',
                          **plot_properties["excluded"], label="excluded")
        h_list.insert(0, h_excl)
        swath = waveform.where(~waveform.exclude_mask)
        h_swath, = ax.plot(swath.ns_20_ku, swath.P[0], ls='', **plot_properties["swath"],
                           label="swath")
        h_list.insert(0, h_swath)
    except KeyError:
        h_all, = ax.plot(waveform.ns_20_ku, waveform.P[0], ls='', **plot_properties["swath"],
                         label="all")
        h_list.insert(0, h_all)
    try:
        poca = waveform.sel(ns_20_ku=waveform.poca_idx)
        h_poca, = ax.plot(poca.ns_20_ku, poca.P[0], ls='', **plot_properties["poca"],
                          label="POCA")
        h_list.insert(0, h_poca)
    except KeyError:
        pass
    ax.set_xlim([waveform.ns_20_ku[0], waveform.ns_20_ku[-1]])
    if logarithmic:
        ax.set_ylim([-200, -100])
    else:
        ax.set_ylim([1e-20, 1e-10])
    ax.set_xlabel("sample number (ns_20_ku)")
    if logarithmic:
        # note that the decibel, here, refers to 1 watt
        ax.set_ylabel("power, decibel")
    else:
        ax.set_ylabel("power, watt")
    ax.legend(handles=h_list)
    ax.set_title(f"id: {waveform.time_20_ku.values[0]}")
    return ax


def dem_transect(waveform, *,
                 ax: plt.Axes = None,
                 line_properties: dict[str, dict[str, any]] = \
                    dict(swath=dict(color="tab:blue",
                                    marker='.',
                                    markersize=5,
                                    linewidth=1),
                         poca=dict(color="tab:green",
                                   marker='o',
                                   markersize=5,
                                   linewidth=1),
                         excluded=dict(color="tab:pink",
                                       marker='x',
                                       markersize=5,
                                       linewidth=1),
                         dem=dict(color="black",
                                  linestyle="solid",
                                  linewidth=0.6,
                                  facecolor="xkcd:ice")
                    ),
                 selected_phase_only: bool = True,
                 dem_file_name_or_path: str = None,
                 ):
    if ax is None:
        ax = plt.subplots()[1]
    dem_reader = get_dem_reader((waveform if dem_file_name_or_path is None else dem_file_name_or_path))
    trans_4326_to_dem_crs = get_4326_to_dem_Transformer(dem_reader)
    sampling_dist = np.arange(-30000, 30000+1, 100)
    num_samples = len(sampling_dist)
    lats, lons = WGS84_ellpsoid.fwd(lons=[waveform.lon_20_ku]*num_samples,
                              lats=[waveform.lat_20_ku]*num_samples,
                              az=[waveform.azimuth+90]*num_samples,
                              dist=sampling_dist)[1::-1]
    xs, ys = trans_4326_to_dem_crs.transform(lats, lons)
    ref_elevs = np.fromiter(dem_reader.sample([(x, y) for x, y in zip(xs, ys)]), "float32")
    ref_elevs = np.where(ref_elevs!=dem_reader.nodata, ref_elevs, np.nan)
    ax.fill_between(sampling_dist, ref_elevs, **line_properties["dem"], label="DEM")
    h_dem = ax.get_legend_handles_labels()[0][0]
    if not selected_phase_only:
        for ph_idx in waveform.phase_wrap_factor.values:
            temp = waveform.sel(phase_wrap_factor=ph_idx)
            temp = temp.where(waveform.ph_idx!=ph_idx).squeeze()
            ax.plot(temp.xph_dists, temp.xph_elevs, '.', c=f"{(.2+.2*np.abs(ph_idx)):.1f}")
    best_phase = waveform.sel(phase_wrap_factor=waveform.ph_idx)#[["xph_dists", "xph_elevs", "exclude_mask"]]
    try:
        excluded = best_phase.where(best_phase.exclude_mask).transpose("ns_20_ku", ...)
        h_excl, = ax.plot(excluded.xph_dists, excluded.xph_elevs, ls='', **line_properties["excluded"],
                          label="excluded")
        swath = best_phase.where(~best_phase.exclude_mask).transpose("ns_20_ku", ...)
        h_swath, = ax.plot(swath.xph_dists, swath.xph_elevs, ls='', **line_properties["swath"], label="swath")
        h_list = [h_swath, h_excl]
    except KeyError:
        h_all, = ax.plot(best_phase.xph_dists, best_phase.xph_elevs, ls='', **line_properties["swath"], label="all")
        h_list = [h_all]
    try:
        poca = best_phase.sel(ns_20_ku=best_phase.poca_idx)
        h_poca, = ax.plot(poca.xph_dists, poca.xph_elevs, ls='', **line_properties["poca"], label="POCA")
        h_list.insert(0, h_poca)
    except KeyError:
        pass
    h_list.append(h_dem)
    ax.legend(handles=h_list)
    ax.set_xlabel("across-track distance to nadir, km")
    ax.set_ylabel("elevation, m")
    ax.set_title(f"id: {waveform.time_20_ku.values[0]}")
    return ax