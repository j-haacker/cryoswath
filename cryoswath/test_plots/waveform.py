import matplotlib.pyplot as plt
import numpy as np

from ..l1b import *
from ..gis import *
from ..misc import *

__all__ = ["dem_transect"]

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
                                  linewidth=0.6)
                    ),
                 selected_phase_only: bool = True):
    if ax is None:
        ax = plt.subplots()[1]
    dem_reader = get_dem_reader(waveform)
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
    h_dem, = ax.plot(sampling_dist, ref_elevs, **line_properties["dem"], label="DEM")
    if not selected_phase_only:
        for ph_idx in waveform.phase_wrap_factor.values:
            temp = waveform.sel(phase_wrap_factor=ph_idx)
            temp = temp.where(waveform.ph_idx!=ph_idx)
            ax.plot(temp.xph_dists, temp.xph_elevs, '.', c=f"{(.2+.2*np.abs(ph_idx)):.1f}")
    best_phase = waveform.sel(phase_wrap_factor=waveform.ph_idx)#[["xph_dists", "xph_elevs", "below_thresholds"]]
    try:
        excluded = best_phase.where(best_phase.below_thresholds)
        h_excl, = ax.plot(excluded.xph_dists, excluded.xph_elevs, ls='', **line_properties["excluded"], label="excluded")
        swath = best_phase.where(~best_phase.below_thresholds)
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
    return ax