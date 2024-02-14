import matplotlib.pyplot as plt
import numpy as np

from ..l1b import *
from ..gis import *
from ..misc import *

__all__ = ["dem_transect"]

def dem_transect(waveform):
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
    ax.plot(sampling_dist, ref_elevs)
    # waveform = tidy_up(waveform, {"xph_dists": "dist_to_grdtrk"})
    best_phase = waveform.sel(phase_wrap_factor=waveform.ph_idx)#[["xph_dists", "xph_elevs", "below_thresholds"]]
    ax.plot(best_phase.xph_dists, best_phase.xph_elevs, '.')
    try:
        ax.plot(best_phase.xph_dists.values[best_phase.below_thresholds.values], best_phase.xph_elevs.values[best_phase.below_thresholds.values], 'rx')
    except KeyError:
        pass
    return ax