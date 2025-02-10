"""cryoswath.l1b module

It mainly contains the L1bData class, that allows to process ESA
CryoSat-2 SARIn L1b data to point elevation estimate (L2 data).
"""

helper_functions = [
    "noise_val",
]

__all__ = [
    "L1bData",
    *helper_functions
]

import fnmatch
import ftplib
import geopandas as gpd
import numbers
import numpy as np
from numpy.typing import ArrayLike
import operator
import os
import pandas as pd
from pyproj import Transformer
import rioxarray as rioxr
from scipy.stats import median_abs_deviation, ttest_ind
import shapely
from threading import Event
import time
from typing import Self
import warnings
import xarray as xr

from .misc import *
from .gis import buffer_4326_shp, \
                 find_planar_crs, \
                 ensure_pyproj_crs, \
                 subdivide_region

__all__ = list()

# requires implicitly rasterio(?), flox(?), dask(?)


def noise_val(vec: ArrayLike) -> float:
    """calculate average noise values for waveform

    Args:
        vec (ArrayLike): First few (well more than 30) samples of power waveform.

    Returns:
        float: Noise power
    """
    # use sufficiently large slices (well more than 6 members)
    n = 30  # slice_thickness
    # iterate over slices: use those of which the average
    # does not significantly differ from previous slices
    # collectively
    for i in range(round(len(vec)/n)-1): # look at first quarter samples
        print(i, np.mean(vec[:(i+2)*n]), np.mean(vec[(i+1)*n:(i+2)*n]), ttest_ind(vec[:(i+1)*n], vec[(i+1)*n:(i+2)*n]))
        if ttest_ind(vec[:(i+1)*n], vec[(i+1)*n:(i+2)*n], equal_var=False).pvalue < 0.001:
            return np.mean(vec[:(i+1)*n])
    return np.mean(vec)


class L1bData(xr.Dataset):
    """Class to wrap functions and properties for L1b data.

    Args to init:
        l1b_filename (str): File to read data from.

        waveform_selection (int | pd.Timestamp | list[int |
            pd.Timestamp] | slice, optional): Waveforms to retrieve data
            from. If none provided, retrieve all data. Defaults to None.

        drop_waveforms_by_flag (dict[str, list], optional):
            Exclude waveform based on flags. Defaults to
            {"flag_mcd_20_ku", [ 'block_degraded', 'blank_block',
            'datation_degraded', 'orbit_prop_error', 'echo_saturated',
            'other_echo_error', 'sarin_rx1_error', 'sarin_rx2_error',
            'window_delay_error', 'agc_error', 'trk_echo_error',
            'echo_rx1_error', 'echo_rx2_error', 'npm_error',
            'power_scale_error']}.

        mask_coherence_gt1 (bool, optional): Defaults to True.

        drop_outside (float, optional): Exclude waveforms where nadir is
            a chosen distance in meters outside of any RGI glacier. If
            None, no waveforms are excluded. Defaults to 30_000.

        coherence_threshold (float, optional): Exclude waveform samples
            with a lower coherence. This choice also affects the
            grouping, start sample for swath processing per waveform,
            and the POCA retrieval. Defaults to 0.6.
            
        power_threshold (tuple, optional): Similar to the coherence
            threshold, but does not affect swath start or POCA
            retrieval. Defaults to ("snr", 10).
    """

    def __init__(self, l1b_filename: str, *,
                 waveform_selection: int|pd.Timestamp|list[int|pd.Timestamp]|slice = None,
                 drop_waveforms_by_flag: dict[str, list] = {"flag_mcd_20_ku": [
                    'block_degraded',
                    'blank_block',
                    'datation_degraded',
                    'orbit_prop_error',
                    'echo_saturated',
                    'other_echo_error',
                    'sarin_rx1_error',
                    'sarin_rx2_error',
                    'window_delay_error',
                    'agc_error',
                    'trk_echo_error',
                    'echo_rx1_error',
                    'echo_rx2_error',
                    'npm_error',
                    'power_scale_error']},
                 mask_coherence_gt1: bool = True,
                 drop_outside: float = 30_000,
                 coherence_threshold: float = 0.6,
                 power_threshold: tuple = ("snr", 10),
                 smooth_phase_difference: bool = True,
                 use_original_noise_estimates: bool = False,
                 dem_file_name_or_path: str = None,
                 ) -> None:
        # ! tbi customize or drop misleading attributes of xr.Dataset
        # currently only originally named CryoSat-2 SARIn files implemented
        assert(fnmatch.fnmatch(l1b_filename, "*CS_????_SIR_SIN_1B_*.nc"))
        patchdicts = [{ "module":       xr.coding.variables,
                        "target":       "_scale_offset_decoding",
                        "replacement":  patched_xr_decode_scaling,
                        "version":      xr.__version__,
                        "rules":        [{  "version":      "2024.3",
                                            "comperator":   operator.lt,
                                            "action":       "skip"},
                                         {  "version":      "2026",
                                            "comperator":   operator.ge,
                                            "action":       "warn"}]},
                      { "module":       xr.coding.times,
                        "target":       "decode_cf_timedelta",
                        "replacement":  patched_xr_decode_tDel,
                        "version":      xr.__version__,
                        "rules":        [{  "version":      "2025",
                                            "comperator":   operator.lt,
                                            "action":       "skip"},
                                         {  "version":      "2025.2",
                                            "comperator":   operator.ge,
                                            "action":       "warn"}]}]
        try:
            with monkeypatch(patchdicts):
                tmp = xr.open_dataset(l1b_filename)#, chunks={"time_20_ku": 256}
        except (OSError, ValueError) as err:
            if isinstance(err, OSError):
                if not err.errno == -101:
                    raise err
                else:
                    warnings.warn(err.strerror+" was raised. Downloading file again.")
            else:
                warnings.warn(str(err)+" was raised. Downloading file again.")
            os.remove(l1b_filename)
            download_single_file(os.path.split(l1b_filename)[-1][19:34])
            with monkeypatch(patchdicts):
                tmp = xr.open_dataset(l1b_filename)
        # at least until baseline E ns_20_ku needs to be made a coordinate
        tmp = tmp.assign_coords(ns_20_ku=("ns_20_ku", np.arange(len(tmp.ns_20_ku))))
        # remove data that will not be used to reduce memory footprint
        for dim in ["time_plrm_01_ku", "time_plrm_20_ku", "nlooks_ku", "space_3d"]:
            if dim in tmp.dims:
                tmp = tmp.drop_dims(dim)
        # first: get azimuth bearing from smoothed incremental azimuths.
        # this needs to be done before dropping part of the recording
        poly3fit_params = np.polyfit(np.arange(len(tmp.time_20_ku)-1), 
                                     WGS84_ellpsoid.inv(lats1=tmp.lat_20_ku[:-1], lons1=tmp.lon_20_ku[:-1],
                                                        lats2=tmp.lat_20_ku[1:], lons2=tmp.lon_20_ku[1:])[0],
                                     3)
        tmp = tmp.assign(azimuth=("time_20_ku", np.poly1d(poly3fit_params)(np.arange(len(tmp.time_20_ku)-.5))%360))
        # waveform selection is meant to be versatile. however the handling seems fragile
        if waveform_selection is not None:
            if not isinstance(waveform_selection, slice) \
            and not isinstance(waveform_selection, list) \
            and not isinstance(waveform_selection, pd.Index):
                waveform_selection = [waveform_selection]
            if (isinstance(waveform_selection, slice) and isinstance(waveform_selection.start, numbers.Integral)) \
                    or isinstance(waveform_selection[0], numbers.Integral):
                tmp = tmp.isel(time_20_ku=waveform_selection)
            else:
                # for compatibility with lower precision timestamps, use backfill or
                # nearest. I prefer nearest because it should also work in cases where
                # the timestamp was rounded instead of floored. for cryosat it should be
                # safe to allow a mismatch of up to +-25 milliseconds (20 Hz).
                tmp = tmp.sel(time_20_ku=waveform_selection, method="nearest",
                              tolerance=np.timedelta64(25, "ms"))
        if mask_coherence_gt1:
            tmp["coherence_waveform_20_ku"] = tmp.coherence_waveform_20_ku.where(tmp.coherence_waveform_20_ku <= 1)
        tmp["power_waveform_20_ku"] = tmp.pwr_waveform_20_ku \
                                       * tmp.echo_scale_factor_20_ku \
                                       * 2**tmp.echo_scale_pwr_20_ku
        if drop_waveforms_by_flag:
            # see available flags using data.flag.attrs["flag_meanings"]
            # print("drop bad. cur buf:", buffer)
            for flag_var, flag_val_list in drop_waveforms_by_flag.items():
                tmp = drop_waveform(tmp, build_flag_mask(tmp[flag_var], flag_val_list))
        if not use_original_noise_estimates:
            # consider noise estimates over periods on the scale of
            # multiple tracking cycles to avoid loss-of-lock issues
            tracking_cycles = 5
            # the implemented algorithm uses a forward and a backward
            # rolling minimum. to work it needs at least twice the
            # window width (however, it is designed for much longer
            # tracks)
            if len(tmp.time_20_ku) > 2*(tracking_cycles*20):
                noise = xr.apply_ufunc(noise_val, tmp.power_waveform_20_ku.isel(ns_20_ku=slice(int(len(tmp.ns_20_ku)/4))), input_core_dims=[["ns_20_ku"]], output_core_dims=[[]], vectorize=True)
                def noise_floor(noise):
                    # construct a lower envelope of the noise values
                    window_size = 5*20 # on the scale of the tracking loop (1 Hz)
                    fwd = noise.rolling(time_20_ku=window_size).min()
                    bwd = noise.isel(time_20_ku=slice(None,None,-1)).rolling(time_20_ku=window_size).min().sortby("time_20_ku")
                    # the upper envelope of the two lower envelope builds
                    # the collective lower envelope
                    upper_envelope = xr.concat([fwd, bwd], "tmp").max("tmp")
                    return upper_envelope.fillna(upper_envelope.max())
                tmp["noise_power_20_ku"] = noise_floor(noise)
        else:
            tmp["noise_power_20_ku"] = tmp.transmit_pwr_20_ku*10**(tmp.noise_power_20_ku/10)
        if drop_outside is not None and drop_outside != False:
            # ! needs to be tidied up:
            # (also: simplify needed?)
            planar_crs = find_planar_crs(lon=tmp.lon_20_ku, lat=tmp.lat_20_ku)
            ground_track_points_4326 = gpd.GeoSeries(gpd.points_from_xy(tmp.lon_20_ku, tmp.lat_20_ku), crs=4326)
            try:
                if isinstance(drop_outside, (int, float)):
                    o2regions = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
                    intersected_o2 = o2regions.geometry.intersects(ground_track_points_4326.union_all(method="unary"))
                    if sum(intersected_o2) == 0:
                        raise IndexError
                    else:
                        o2codes = o2regions.loc[intersected_o2,"o2region"].values
                    o2region_complexes = []
                    for o2 in np.unique(o2codes):
                        if o2 != "05-01": # Greenland periphery is too large
                            o2region_complexes.append(load_o2region(o2))
                        else: # cut into 10 subregions, append if crossed
                            # !tbi: instead of using the arbitrary chunks, use the custom subregions
                            # 05-11--05-15 (added in commit 2265523)
                            for grnlnd_part in subdivide_region(load_o2region("05-01"), lat_bin_width_degree=4.5,
                                                                lon_bin_width_degree=4.5):
                                if buffer_4326_shp(grnlnd_part.union_all(method="coverage").envelope, drop_outside)\
                                        .intersects(ground_track_points_4326.union_all(method="unary")):
                                    o2region_complexes.append(grnlnd_part)
                    # below, using geopandas as shapely wrapper for readability
                    buffered_complexes = gpd.GeoSeries(
                        buffer_4326_shp(pd.concat(o2region_complexes).union_all(method="coverage"), drop_outside), crs=4326
                        ).to_crs(planar_crs).clip_by_rect(*ground_track_points_4326.to_crs(planar_crs).total_bounds)\
                        .to_crs(4326).make_valid().iloc[0]
                else:
                    buffered_complexes = drop_outside
                retain_indeces = ground_track_points_4326.intersects(buffered_complexes)
                tmp = tmp.isel(time_20_ku=retain_indeces[retain_indeces].index)
            except IndexError:
                warnings.warn("No waveforms left on glacier. Proceeding with empty dataset.")
                tmp = tmp.isel(time_20_ku=[])
        tmp = tmp.assign_attrs(coherence_threshold=coherence_threshold,
                               power_threshold=power_threshold,
                               smooth_phase_difference=smooth_phase_difference)
        # find and store POCAs and swath-starts
        tmp = append_poca_and_swath_idxs(tmp)
        # use lowpass-filtered phase difference at POCA
        tmp = append_smoothed_complex_phase(tmp)
        tmp["ph_diff_waveform_20_ku"] = xr.where(
            tmp.ns_20_ku==tmp.poca_idx,
            xr.apply_ufunc(np.angle, tmp.ph_diff_complex_smoothed),
            tmp.ph_diff_waveform_20_ku
        )
        # add potential phase wrap factor for later use
        tmp = tmp.assign_coords({"phase_wrap_factor": np.arange(-3, 4)})
        super().__init__(data_vars=tmp.data_vars, coords=tmp.coords, attrs=tmp.attrs)
    
    def append_ambiguous_reference_elevation(self, dem_file_name_or_path: str = None):
        # !! This function causes much of the computation time. I suspect that
        # sparse memory accessing can be minimized with some tricks. However,
        # first tries sorting the spatial data, took even (much) longer.
        if not "xph_lats" in self.data_vars:
            self = self.locate_ambiguous_origin()
        # ! tbi: auto download ref dem if not present
        with get_dem_reader((self if dem_file_name_or_path is None else dem_file_name_or_path)) as dem_reader:
            trans_4326_to_dem_crs = Transformer.from_crs('EPSG:4326', dem_reader.crs)
            x, y = trans_4326_to_dem_crs.transform(self.xph_lats, self.xph_lons)
            self = self.assign(xph_x=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), x), 
                               xph_y=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), y))
            self.attrs.update({"CRS": ensure_pyproj_crs(dem_reader.crs)})
            # ! huge improvement potential: instead of the below, rasterio.sample could be used
            # [edit] use postgis
            try:
                ref_dem = rioxr.open_rasterio(dem_reader).rio.clip_box(np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)).squeeze()
            except rioxr.exceptions.NoDataInBounds:
                warnings.warn(f"couldn't find ref dem data in box: {np.nanmin(x)}, {np.nanmin(y)}, {np.nanmax(x)}, {np.nanmax(y)}\nouter lat lon coords: {self.lat_20_ku.values[[0,-1]]}, {self.lon_20_ku.values[[0,-1]]}")
                raise
            self["xph_ref_elevs"] = ref_dem.sel(x=self.xph_x, y=self.xph_y, method="nearest")
        # rasterio suggests sorting like `for ind in np.lexsort([y, x]): rv.append((x[ind], y[ind]))`
        # sort_key = np.lexsort([y, x])
        # planar_coords = zip(x[sort_key], y[sort_key])
        # ref_elev_vector = np.fromiter(dem_reader.sample(planar_coords), "float32")[sort_key.argsort()]
        # return self.assign(xph_ref_elevs=(self.xph_lats.dims,
        #                                   np.reshape(ref_elev_vector,
        #                                              self.xph_lats.shape)))
        return self

    def append_best_fit_phase_index(self, best_column: callable = None) -> Self:
        """Resolve phase difference ambiguity

        The phase difference is ambiguous and only know except for a multiple
        of 2 pi. This method finds the best fitting factor of 2 pi wrt. a
        digital elevation model (DEM). By default, the summed distance to the
        DEM per group is minimized.

        Args:
            best_column (callable, optional): Function that takes a k*n matrix of
                difference to the DEM as first argument, where k are the number
                of group members (waveform samples) and n the number of possible
                wrapping factors. The function needs to return the chosen index
                along the second axis. Visit the source code to get a template
                for an excepted function. Defaults to None.

        Returns:
            L1bData
        """
        # ! Implement opt-out or/and grouping alternatives
        # before locating echos, find groups because also phase is unwrapped
        if not "group_id" in self.data_vars:
            self = self.tag_groups()
            # it makes sense to always unwrap the phases immediately after finding
            # the groups. assigning the best fitting indices otherwise messes up
            # your data
            self = self.unwrap_phase_diff()
        if not "xph_elev_diffs" in self.data_vars:
            self = self.append_elev_diff_to_ref()
        self = self.assign(ph_idx=(("time_20_ku", "ns_20_ku"),
                                   np.empty((len(self.time_20_ku), len(self.ns_20_ku)), dtype="int")))
        if best_column is None:
            def best_column(elev_diff):
                return np.argmin(np.abs(np.median(elev_diff, axis=0))**2+median_abs_deviation(elev_diff, axis=0)**2)
        def find_group_ph_idx(elev_diff, group_ids):
            out = np.zeros_like(group_ids)
            for i in nan_unique(group_ids):
                mask = group_ids == i
                out[mask] = best_column(elev_diff[mask,:])-len(self.phase_wrap_factor)//2
            return out
        self["ph_idx"] = xr.apply_ufunc(find_group_ph_idx, 
                                        self.xph_elev_diffs, 
                                        self.group_id,
                                        input_core_dims=[["ns_20_ku", "phase_wrap_factor"], ["ns_20_ku"]],
                                        output_core_dims=[["ns_20_ku"]])
        self["ph_idx"] = xr.where(self.group_id.isnull(), np.abs(self.xph_elev_diffs).idxmin("phase_wrap_factor"), self.ph_idx)
        return self
    
    def append_elev_diff_to_ref(self):
        if not "xph_ref_elevs" in self.data_vars:
            self = self.append_ambiguous_reference_elevation()
        self["xph_elev_diffs"] = (self.xph_elevs-self.xph_ref_elevs)
        return self
    
    @classmethod
    def from_id(cls, track_id: str|pd.Timestamp, **kwargs) -> Self:
        track_id = pd.to_datetime(track_id)
        # edge cases with exactly 0 nanoseconds may fail. however, since this is
        # only relevant for detail inspection, edge cases are ignored
        if track_id.nanosecond != 0:
            kwargs=dict(waveform_selection=track_id)
            # file name list as look up table
            full_file_names = load_cs_full_file_names(update="no")
            idx_loc = full_file_names.index.get_indexer([track_id], method="pad")[0]
            track_id = full_file_names.index[idx_loc]
        l1b_data_dir = os.path.join(data_path, "L1b", track_id.strftime(f"%Y{os.path.sep}%m"))
        track_id = cs_time_to_id(track_id)
        if os.path.isdir(l1b_data_dir):
            for file_name in os.listdir(l1b_data_dir):
                if fnmatch.fnmatch(file_name, "*CS_????_SIR_SIN_1B_*") \
                and os.path.split(file_name)[-1][19:34] == track_id \
                and file_name.endswith(".nc"):
                    return cls(os.path.join(l1b_data_dir, file_name), **kwargs)
        return cls(download_single_file(track_id), **kwargs)

    def get_rgi_o2(self) -> str:
        """Finds RGIv7 o2 region that contains the track's central lat,
        lon.

        Returns:
            str: RGI v7 `long_code`
        """
        rgi_o2_gpdf = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
        return rgi_o2_gpdf[rgi_o2_gpdf.contains(
                gpd.points_from_xy(self.lon_20_ku, self.lat_20_ku, crs=4326).unary_all(method="coverage").centroid
            )].long_code.values[0]

    def phase_jump(self):
        ph_diff_diff = self.ph_diff_complex_smoothed.diff("ns_20_ku")
        # ! implement choosing tolerance
        ph_diff_diff_tolerance = .1
        jump_mask =  np.logical_or(np.abs(ph_diff_diff) > ph_diff_diff_tolerance,
                                np.abs(ph_diff_diff).rolling(ns_20_ku=2).sum() > 2* 0.8*ph_diff_diff_tolerance)
        if not "exclude_mask" in self.data_vars:
            self = append_exclude_mask(self)
        return xr.where(self.exclude_mask.sel(ns_20_ku=jump_mask.ns_20_ku), False, jump_mask)

    def phase_outlier(self, tol: float|None = None):
        # inputs have to be complex unit vectors
        # if no tol provided calc equivalent of 300 m at nadir
        if tol == None:
            temp_x_width = 300 # [m] allow ph_diff to jump by this value (roughly)
            temp_H = 720e3 # [m] rough altitude of CS2
            # 0s below: set to an arbitrary off nadir angle at which the x_width should actually have the defined value
            tol = (np.arctan(np.tan(np.deg2rad(0))+temp_x_width/temp_H)-np.deg2rad(0)) \
                   * 2*np.pi / np.tan(speed_of_light/Ku_band_freq/antenna_baseline)
        # ph_diff_tol is small, so approx equal to secant length
        return np.abs(np.exp(1j*self.ph_diff_waveform_20_ku) - self.ph_diff_complex_smoothed) > tol

    # ! rename to something like retrieve_ambiguous_origins
    def locate_ambiguous_origin(self):
        """Calculates all "possible" echo origins.

        Adds for the 7 look angles `xph_thetas` the variables xph_lats,
        xph_lons, xph_elevs, and xph_dists.

        Returns:
            Dataset: l1b_data including the calculated coordinates.
        """
        # Calculate normal distance: position on ellipsoid surface <--> major axis
        r_N = WGS84_ellpsoid.a/np.sqrt(1-WGS84_ellpsoid.es*np.sin(np.deg2rad(self.lat_20_ku))**2)
        # Add satellite height
        r_cs2 = r_N+self.alt_20_ku
        # Calculate distance: satellite <--> echo origin
        range_to_scat = self.ref_range()+(self.ns_20_ku-512)*sample_width
        theta = np.arcsin(-(self.ph_diff_waveform_20_ku + self.phase_wrap_factor*2*np.pi)
                          * (speed_of_light/Ku_band_freq)/(2*np.pi*antenna_baseline)) \
                - np.deg2rad(self.off_nadir_roll_angle_str_20_ku)
        # Calculate distance: echo origin <--> major axis (from scalar product)
        r_x = np.sqrt( range_to_scat**2 + r_cs2**2 - (2*range_to_scat*r_cs2*np.cos(theta)) )
        dist_off_groundtrack = r_N*np.arctan(range_to_scat*np.sin(theta)/(r_cs2-range_to_scat*np.cos(theta)))
        lons, lats = WGS84_ellpsoid.fwd(lons=self.lon_20_ku.expand_dims({"ns_20_ku": self.ns_20_ku.size,
                                                                   "phase_wrap_factor": self.phase_wrap_factor.size}, [-2, -1]),
                                  lats=self.lat_20_ku.expand_dims({"ns_20_ku": self.ns_20_ku.size,
                                                                   "phase_wrap_factor": self.phase_wrap_factor.size}, [-2, -1]),
                                  az=self.azimuth.expand_dims({"ns_20_ku": self.ns_20_ku.size,
                                                               "phase_wrap_factor": self.phase_wrap_factor.size}, [-2, -1])+90,
                                  dist=dist_off_groundtrack)[:2]
        return self.assign(xph_lons=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), lons),
                           xph_lats=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), lats),
                           # Assuming the local ellipsoid radius changes slowly:
                           xph_elevs=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), (r_x - r_N).transpose("time_20_ku", "ns_20_ku", "phase_wrap_factor").values),
                           xph_thetas=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), theta.transpose("time_20_ku", "ns_20_ku", "phase_wrap_factor").values),
                           xph_dists=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), dist_off_groundtrack.transpose("time_20_ku", "ns_20_ku", "phase_wrap_factor").values))
    
    def ref_range(self) -> xr.DataArray:
        """Calculate distance to center of range window.

        Returns:
            xr.DataArray: Reference ranges.
        """
        # make property?
        corrections = self.mod_dry_tropo_cor_01 \
                      + self.mod_wet_tropo_cor_01 \
                      + self.iono_cor_gim_01 \
                      + self.pole_tide_01 \
                      + self.solid_earth_tide_01 \
                      + self.load_tide_01
        return self.window_del_20_ku/np.timedelta64(1, 's') / 2 * speed_of_light \
               + np.interp(self.time_20_ku, self.time_cor_01, corrections)
    
    def tag_groups(self) -> Self:
        """Identifies and tags wafeform sample groups.

        Returns:
            Self: l1b_ds.
        """
        # print("debug tag groups 0", flush=True)
        phase_outlier = self.phase_outlier()
        # print("debug tag groups 0.1", flush=True)
        ignore_mask = (self.exclude_mask + phase_outlier) != 0
        gap_separator = ignore_mask.rolling(ns_20_ku=3).sum() == 3
        # print("debug tag groups 0.2", flush=True)
        any_separator = np.logical_or(*xr.align(self.phase_jump(), gap_separator, join="outer"))
        # print("debug tag groups 0.2.1", flush=True)
        rising_edge_per_waveform_counter = (any_separator.astype('int32').diff("ns_20_ku")==-1).cumsum("ns_20_ku") + 1
        # print(rising_edge_per_waveform_counter)
        # print("debug tag groups 0.3", flush=True)
        group_tags = rising_edge_per_waveform_counter \
            + xr.DataArray(data=np.arange(len(self.time_20_ku))*len(self.ns_20_ku), dims="time_20_ku")
        group_tags = xr.align(group_tags, self.power_waveform_20_ku, join="right")[0].where(~ignore_mask)
        # print("debug tag groups 0.4", flush=True)
        def filter_small_groups(group_ids):
            out = group_ids
            for i in nan_unique(group_ids):
                mask = group_ids == i
                if mask.sum() < 3:
                    out[mask] = 0
            return out
        # print("debug tag groups 1", flush=True)
        group_tags = xr.apply_ufunc(filter_small_groups,
                       group_tags,
                       input_core_dims=[["ns_20_ku"]],
                       output_core_dims=[["ns_20_ku"]])
        group_tags = group_tags.where(group_tags != 0)
        self["group_id"] = group_tags
        return self

    def to_l2(self, out_vars: list|dict = None, *,
                    retain_vars: list|dict = None,
                    swath_or_poca: str = "swath",
                    group_best_column_func: callable = None,
                    **kwargs) -> gpd.GeoDataFrame:
        """Converts l1b data to l2 data (point elevations).

        Args:
            out_vars (list | dict, optional): Return values. If none provided,
                returns time, x, y, height, reference elevation, and difference
                wrt. reference. Provide a dictionary to assign custom names.
                Defaults to None.
            retain_vars (list | dict, optional): Additional to `out_vars`.
                Defaults to None.
            swath_or_poca (str, optional): Either "swath", "poca", or "both".
                Decides what data is returned. Defaults to "swath".
            group_best_column_func (callable, optional): Optimization function to
                resolve phase difference ambiguity. View
                :func:`append_best_fit_phase_index` for details.

        Raises:
            ValueError: If `swath_or_poca` cannot be interpreted.

        Returns:
            gpd.GeoDataFrame: Elevation estimates and requested variables. If
            `swath_or_poca` is "both", a tuple with separate tables is
            returned.
        """
        if len(self.time_20_ku) == 0:
            if swath_or_poca == "both":
                return gpd.GeoDataFrame(), gpd.GeoDataFrame()
            else:
                return gpd.GeoDataFrame()
        if out_vars is None:
            out_vars = dict(time_20_ku="time",
                            xph_x="x",
                            xph_y="y",
                            xph_elevs="height",
                            xph_ref_elevs="h_ref",
                            xph_elev_diffs="h_diff")
        # implicitly test whether data was processed. if not, do so
        if not "ph_idx" in self.data_vars:
            self = self.append_best_fit_phase_index(group_best_column_func)
        if isinstance(out_vars, dict):
            self = self.drop_vars(list(out_vars.values()), errors="ignore")
            self = self.rename_vars(out_vars)
            out_vars = list(out_vars.values())
        if isinstance(retain_vars, dict):
            self = self.drop_vars(list(retain_vars.values()), errors="ignore")
            self = self.rename_vars(retain_vars)
            retain_vars = list(retain_vars.values())
        elif retain_vars is None:
            retain_vars = []
        if swath_or_poca == "swath":
            tmp = self[out_vars+retain_vars].where(~self.exclude_mask)\
                                               .sel(phase_wrap_factor=self.ph_idx)\
                                               .dropna("time_20_ku", how="all")
        elif swath_or_poca == "poca":
            waveforms_with_poca = self.time_20_ku[~self.poca_idx.isnull()]
            if len(waveforms_with_poca) == 0:
                return gpd.GeoDataFrame()
            tmp = self[out_vars+retain_vars+["ph_idx"]].sel(time_20_ku=waveforms_with_poca)\
                                                          .sel(ns_20_ku=self.poca_idx[~self.poca_idx.isnull()])
            tmp = tmp[out_vars+retain_vars].sel(phase_wrap_factor=tmp.ph_idx).dropna("time_20_ku", how="all")
        elif swath_or_poca == "both":
            swath = self.to_l2(out_vars, retain_vars=retain_vars, swath_or_poca="swath", **kwargs)
            poca = self.to_l2(out_vars, retain_vars=retain_vars, swath_or_poca="poca", **kwargs)
            return swath, poca
        else:
            raise ValueError(f"You provided \"swath_or_poca={swath_or_poca}\". Choose \"swath\", \"poca\",",
                             "or \"both\".")
        drop_coords = [coord for coord in tmp.coords if coord not in ["time", "sample"]]
        from . import l2 # can't be in preamble as this would lead to circularity
        # ! dropped .squeeze() below to handle issue #19. not sure about 2nd degree consequences.
        # l2_data = l2.from_processed_l1b(tmp.squeeze().drop_vars(drop_coords), **kwargs)
        l2_data = l2.from_processed_l1b(tmp.drop_vars(drop_coords), **kwargs)
        return l2_data

    def unwrap_phase_diff(self) -> Self:
        """Replaces phase difference by unwrapped version.

        Unwrapping is done per group of waveform samples.

        Returns:
            Self: l1b_ds.
        """

        def unwrap(ph_diff, group_ids):
            out = ph_diff
            for i in nan_unique(group_ids):
                mask = group_ids == i
                out[mask] = np.unwrap(ph_diff[mask])
            return out
        
        if self.attrs["smooth_phase_difference"]: #
            self["ph_diff_waveform_20_ku"] = xr.where(self.ph_diff_complex_smoothed.isnull(),
                                                        self.ph_diff_waveform_20_ku,
                                                        xr.apply_ufunc(np.angle, self.ph_diff_complex_smoothed))
            
        self["ph_diff_waveform_20_ku"] = xr.apply_ufunc(unwrap, 
                                                        self.ph_diff_waveform_20_ku, 
                                                        self.group_id,
                                                        input_core_dims=[["ns_20_ku"], ["ns_20_ku"]],
                                                        output_core_dims=[["ns_20_ku"]])
        return self


# helper functions ####################################################
    
def append_exclude_mask(cs_l1b_ds: "L1bData") -> "L1bData":
    """Adds mask indicating samples below threshold.

    Waveform samples that don't fulfill power and/or coherence requirements
    are flagged. The thresholds have to be included in the provided
    dataset. By default, they are assigned on creation.

    Args:
        cs_l1b_ds (l1b_data): Input data.

    Returns:
        l1b_data: Data including mask.
    """
    # for now require tuple. could be some auto recognition in future.
    assert(isinstance(cs_l1b_ds.power_threshold, tuple))
    # only signal-to-noise-ratio implemented
    assert(cs_l1b_ds.power_threshold[0] == "snr")
    power_threshold = cs_l1b_ds.noise_power_20_ku*cs_l1b_ds.power_threshold[1]
    cs_l1b_ds["exclude_mask"] = \
        np.logical_or(cs_l1b_ds.power_waveform_20_ku < power_threshold,
                      cs_l1b_ds.coherence_waveform_20_ku < cs_l1b_ds.coherence_threshold)
    return cs_l1b_ds
__all__.append("append_exclude_mask")
    

def append_poca_and_swath_idxs(cs_l1b_ds: "L1bData") -> "L1bData":
    """Adds indices for estimated POCA and begin of swath.

    Args:
        cs_l1b_ds (l1b_data): Input data.

    Returns:
        l1b_data: Data including mask.
    """
    if len(cs_l1b_ds.time_20_ku) == 0:
        return cs_l1b_ds.assign(swath_start=(("time_20_ku"), []),
                                poca_idx=(("time_20_ku"), []),
                                exclude_mask=(("time_20_ku", "ns_20_ku"),
                                              np.empty_like(cs_l1b_ds.power_waveform_20_ku)))
    # ! performance improvement potential
    # should be possible to accelerate with numba
    def find_poca_idx_and_swath_start_idx(smooth_coh, coh_thr):
        # if smooth coherence exceeds threshold in the first 10 m, its
        # unreasonable to assume that the tracking loop did not fail
        # (the POCA may have been before the waveform even starts, but
        # we can't tell).
        poca_idx = np.argmax(smooth_coh>coh_thr)
        if poca_idx < int(10/sample_width):
            # I opted for nan if no poca for transparency. this requires dtype float and is slower
            return np.nan, 0
        # poca expected 10 m after coherence exceeds threshold (no solid basis)
        poca_idx = np.argmax(smooth_coh[poca_idx:poca_idx+int(10/sample_width)])+poca_idx
        try:
            swath_start = poca_idx + int(5/sample_width)
            diff_smooth_coh = np.diff(smooth_coh[swath_start:swath_start+int(50/sample_width)])
            # swath can safest be used after the coherence dip
            swath_start = np.argmax(diff_smooth_coh[np.argmax(np.abs(diff_smooth_coh)>.001):]>0) + swath_start
        # if swath doesn't start in range window, just indeed set the index behind last element
        except ValueError:
            swath_start = len(smooth_coh)
        return float(poca_idx), swath_start
    cs_l1b_ds[["poca_idx", "swath_start"]] = xr.apply_ufunc(
        find_poca_idx_and_swath_start_idx,
        gauss_filter_DataArray(cs_l1b_ds.coherence_waveform_20_ku, "ns_20_ku", 35, 35),
        kwargs=dict(coh_thr=cs_l1b_ds.coherence_threshold),
        input_core_dims=[["ns_20_ku"]],
        output_core_dims=[[], []],
        vectorize=True)
    if "exclude_mask" not in cs_l1b_ds.data_vars:
        cs_l1b_ds = append_exclude_mask(cs_l1b_ds)
    cs_l1b_ds["exclude_mask"] = xr.where(cs_l1b_ds.ns_20_ku<cs_l1b_ds.swath_start, True, cs_l1b_ds.exclude_mask)
    return cs_l1b_ds
__all__.append("append_poca_and_swath_idxs")
    

def append_smoothed_complex_phase(cs_l1b_ds: "L1bData") -> "L1bData":
    cs_l1b_ds["ph_diff_complex_smoothed"] = gauss_filter_DataArray(np.exp(1j*cs_l1b_ds.ph_diff_waveform_20_ku),
                                                                dim="ns_20_ku", window_extent=21, std=5)
    return cs_l1b_ds
__all__.append("append_smoothed_complex_phase")


def build_flag_mask(cs_l1b_flag: xr.DataArray, flag_val_list: list) -> xr.DataArray:
    """Function returns a waveform mask based on flag values.

    This function can handle two types of flags: those that take the form
    of a checklist with multiple allowed ticks, and those that indicate
    one of more possible selections.

    It is designed for CryoSat-2 SARIn L1b Baseline D or E data and
    relies on an attribute "flag_masks" or "flag_values". For CRISTAL or
    if the attributes change, this function needs an update.

    Args:
        cs_l1b_flag (xr.DataArray): L1bData flag variable.
        flag_val_list (list, optional): List of flag values to mask.

    Returns:
        xr.DataArray: Mask that is True where flag matched provided list.
    """
    if "flag_masks" in cs_l1b_flag.attrs:
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=np.log2(np.abs(cs_l1b_flag.attrs["flag_masks"].astype("int64")
                                                        )).astype("int")).sort_index()
        def flag_func(int_code: int) -> bool:
            for i, b in enumerate(reversed(bin(int_code)[2:])):
                if b == "0": continue
                try:
                    if flag_dictionary.loc[i] in flag_val_list:
                        return True
                except KeyError:
                    print("Flag not found in attributes! Pointing to a bug or an issue in the data.")
                    raise
            return False
    elif "flag_values" in cs_l1b_flag.attrs:
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=cs_l1b_flag.attrs["flag_values"])
        def flag_func(int_code: int):
            return flag_dictionary.loc[int_code] in flag_val_list
    else:
        raise NotImplementedError
    return xr.apply_ufunc(np.vectorize(flag_func), cs_l1b_flag.astype(int), dask="allowed")
__all__.append("build_flag_mask")


# ! name is not intuitive
def download_wrapper(region_of_interest: str|shapely.Polygon = None,
                   start_datetime: str|pd.Timestamp = "2010",
                   end_datetime: str|pd.Timestamp = "2035", *,
                   buffer_region_by: float = None,
                   track_idx: pd.DatetimeIndex|str = None,
                   stop_event: Event = None,
                   n_threads: int = 8,
                   #baseline: str = "latest",
                   ) -> int:
    """Download ESA's L1b product.

    Args:
        region_of_interest (str | shapely.Polygon, optional): Provide a RGI
            identifier or lon/lat polygon to subset downloaded data.
            Defaults to None.
        start_datetime (str | pd.Timestamp, optional): Defaults to "2010".
        end_datetime (str | pd.Timestamp, optional): Defaults to "2035".
        buffer_region_by (float, optional): Use a buffer in meter around
            provided region (also RGI identifier). Defaults to None.
        track_idx (pd.DatetimeIndex | str, optional): Download only tracks
            at known times. Defaults to None.
        stop_event (Event, optional): Define when to terminate threads.
            Defaults to None.
        n_threads (int, optional): Number of download threads. Defaults to 8.

    Returns:
        int: 0 on success, 1 on graceful exit after error, and 2 on being
        aborted.
    """
    if track_idx is None:
        start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
        track_idx = load_cs_ground_tracks(region_of_interest, start_datetime, end_datetime, buffer_region_by=buffer_region_by).index
    else:
        track_idx = track_idx.sort_values()
        start_datetime, end_datetime = track_idx[[0,-1]]
    if stop_event is None:
        stop_event = Event()
    task_queue = request_workers(download_files, n_threads)
    months = pd.date_range(pd.offsets.MonthBegin().rollback(start_datetime.normalize()), end_datetime, freq="MS")
    for month in months:
        # print(month)
        # print(track_idx.normalize()+pd.DateOffset(day=1))
        idx_selection = track_idx[track_idx.normalize()+pd.DateOffset(day=1)==month]
        task_queue.put((idx_selection, stop_event))
    # wait for threads to finish
    try:
        task_queue.join()
    except:
        stop_event.set()
        with task_queue.mutex:
            task_queue.queue.clear()
        print("Aborting download because error occured. This may have been an interrupt.")
        for i in range(3):
            time.sleep(10)
            if task_queue.empty():
                print("Closed all download threads. Likely not all files were downloaded.")
                return 1
        print("Forcibly shutting down all download threads. In worst case, leads to fractured nc-files.")
        return 2
    else:
        print("All downloads finished.")
        return 0
__all__.append("download_wrapper")


def download_files(track_idx: pd.DatetimeIndex|str,
                   stop_event: Event = None,
                   #baseline: str = "latest",
                   ):
    year_month_str_list = track_idx.strftime(f"%Y{os.path.sep}%m").unique()
    for year_month_str in year_month_str_list:
        print("scanning", year_month_str, end=" ")
        if stop_event is not None and stop_event.is_set():
            return
        try:
            currently_present_files = [x[19:] for x in os.listdir(os.path.join(l1b_path, year_month_str))]
        except FileNotFoundError:
            os.makedirs(os.path.join(l1b_path, year_month_str))
            currently_present_files = []
        with ftp_cs2_server(timeout=120) as ftp:
            try:
                ftp.cwd("/SIR_SIN_L1/"+year_month_str)
            except ftplib.error_perm:
                warnings.warn("Directory /SIR_SIN_L1/"+year_month_str+" couldn't be accessed.")
                continue
            for remote_file in ftp.nlst():
                if stop_event is not None and stop_event.is_set():
                    return
                if remote_file[-3:] == ".nc" \
                        and pd.to_datetime(remote_file[19:34]) in track_idx \
                        and remote_file[19:] not in currently_present_files:
                    local_path = os.path.join(l1b_path, year_month_str, remote_file)
                    try:
                        with open(local_path, "wb") as local_file:
                            print("downloading", remote_file)
                            # [enhancement] use `binary_cache` as buffer instead of removing on fail
                            ftp.retrbinary("RETR "+remote_file, local_file.write)
                    except:
                        print("download failed for", remote_file)
                        if os.path.isfile(local_path):
                            os.remove(local_path)
                        raise
    print("finished downloading tracks for months:\n", year_month_str_list)
__all__.append("download_files")


def download_single_file(track_id: str) -> str:
    # currently only CryoSat-2
    retries = 10
    while retries > 0:
        try:
            with ftp_cs2_server() as ftp:
                ftp.cwd("/SIR_SIN_L1/"+pd.to_datetime(track_id).strftime("%Y/%m"))
                for remote_file in ftp.nlst():
                    if remote_file[-3:] == ".nc" \
                    and remote_file[19:34] == track_id:
                        local_path = os.path.join(data_path, "L1b", pd.to_datetime(track_id).strftime("%Y/%m"))
                        if not os.path.isdir(local_path):
                            os.makedirs(local_path)
                        local_path = os.path.join(local_path, remote_file)
                        try:
                            with open(local_path, "wb") as local_file:
                                print("downloading "+remote_file)
                                ftp.retrbinary("RETR "+remote_file, local_file.write)
                                return local_path
                        except:
                            print("download failed for", remote_file)
                            if os.path.isfile(local_path):
                                os.remove(local_path)
                            raise
                print(f"File for id {track_id} couldn't be found in remote dir {ftp.pwd()}.")
                # ! should this raise an error?
                raise FileNotFoundError()
        except ftplib.error_temp as err:
            print(str(err), f"raised. Retrying to download file with id {track_id} in 10 s for the {11-retries}. time.")
            time.sleep(10)
            retries -= 1
__all__.append("download_single_file")


def drop_waveform(cs_l1b_ds, time_20_ku_mask):
    """Use mask along time dim to drop waveforms.

    Args:
        time_20_ku_mask (1-dim bool): Mask: drop where True.

    Returns:
        xr.Dataset or DataArray: Input dataset without marked waveforms.
    """
    return cs_l1b_ds.sel(time_20_ku=cs_l1b_ds.time_20_ku[~time_20_ku_mask])
__all__.append("drop_waveform")


# left here for improvement ideas
# def choose_group_phase_wrap(waveform):
#     # this should be possible for all waveforms in parallel (see below). However, this takes much longer for some
#     # reason. Check again, when using dask (looks like a sparse memory accessing issue).
#     # ds["ph_idx"][~ds.group_id.isnull()] = ds.xph_swath_h_diff.groupby(ds.group_id).map(lambda x: x.ns_20_ku*0+x.mean("stacked_time_20_ku_ns_20_ku").idxmin("phase_wrap_factor"))
#     return waveform.xph_elev_diffs.groupby(waveform.group_id).map(lambda x: x.ns_20_ku*0+x.mean("ns_20_ku").idxmin("phase_wrap_factor"))


__all__ = sorted(__all__)
