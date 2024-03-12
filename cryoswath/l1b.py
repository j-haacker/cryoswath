from dateutil.relativedelta import relativedelta
import fnmatch
import ftplib
import geopandas as gpd
import numbers
import numpy as np
import os
import pandas as pd
from pyproj import Transformer
import rasterio as rio
import rioxarray as rioxr
from scipy.spatial.transform import Rotation
import shapely
import warnings
import xarray as xr

from . import gis
from .misc import *

__all__ = list()

# requires implicitly rasterio(?), flox(?), dask(?)

class l1b_data(xr.Dataset):
    # for now, only CryoSat-2 implemented

    __all__ = list()

    def __init__(self, l1b_filename: str, *,
                 waveform_selection: int|pd.Timestamp|list[int|pd.Timestamp]|slice = None,
                 drop_bad_waveforms: bool = True,
                 mask_coherence_gt1: bool = True,
                 drop_non_glacier_areas: bool = True,
                 coherence_threshold: float = 0.6,
                 power_threshold: tuple = ("snr", 10),
                 ) -> None:
        # ! tbi customize or drop misleading attributes of xr.Dataset
        # currently only originally named CryoSat-2 SARIn files implemented
        assert(fnmatch.fnmatch(l1b_filename, "*CS_????_SIR_SIN_1B_*.nc"))
        tmp = xr.open_dataset(l1b_filename)#, chunks={"time_20_ku": 256}
        # at least until baseline E ns_20_ku needs to be made a coordinate
        tmp = tmp.assign_coords(ns_20_ku=("ns_20_ku", np.arange(len(tmp.ns_20_ku))))
        # first: get azimuth bearing from smoothed incremental azimuths.
        # this needs to be done before dropping part of the recording
        poly3fit_params = np.polyfit(np.arange(len(tmp.time_20_ku)-1), 
                                     WGS84_ellpsoid.inv(lats1=tmp.lat_20_ku[:-1], lons1=tmp.lon_20_ku[:-1],
                                                        lats2=tmp.lat_20_ku[1:], lons2=tmp.lon_20_ku[1:])[0],
                                     3)
        tmp = tmp.assign(azimuth=("time_20_ku", np.poly1d(poly3fit_params)(np.arange(len(tmp.time_20_ku)-.5))%360))
        # waveform selection is meant to be versatile. however the handling seems fragile
        if waveform_selection is not None:
            if not isinstance(waveform_selection, slice) and len(waveform_selection) == 1:
                waveform_selection = [waveform_selection]
            if (isinstance(waveform_selection, slice) and isinstance(waveform_selection.start, numbers.Integral)) \
                    or isinstance(waveform_selection[0], numbers.Integral):
                tmp = tmp.isel(time_20_ku=waveform_selection)
            else:
                # for compatibility with lower precision timestamps, use backfill or
                # nearest. I prefer nearest because it should also work in cases where
                # the timestmap was rounded instead of floored. for cryosat it should be
                # safe to allow a mismatch of up to +-25 milliseconds (20 Hz).
                tmp = tmp.sel(time_20_ku=waveform_selection, method="nearest",
                              tolerance=np.timedelta64(25, "ms"))
        if mask_coherence_gt1:
            tmp["coherence_waveform_20_ku"] = tmp.coherence_waveform_20_ku.where(tmp.coherence_waveform_20_ku <= 1)
        tmp["power_waveform_20_ku"] = tmp.pwr_waveform_20_ku \
                                       * tmp.echo_scale_factor_20_ku \
                                       * 2**tmp.echo_scale_pwr_20_ku
        if drop_bad_waveforms:
            # see available flags using data.flag.attrs["flag_meanings"]
            # print("drop bad. cur buf:", buffer)
            tmp = drop_waveform(tmp, build_flag_mask(tmp.flag_mcd_20_ku, [
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
                'power_scale_error',
            ]))
        if drop_non_glacier_areas:
            # ! needs to be tidied up:
            # (also: simplify needed?)
            buffered_points = gpd.GeoSeries(gpd.points_from_xy(tmp.lon_20_ku, tmp.lat_20_ku), crs=4326)\
                .to_crs(3413).buffer(30_000).simplify(5_000).to_crs(4326)
            o2regions = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
            try:
                intersected_o2 = o2regions.geometry.intersects(shapely.box(*buffered_points.total_bounds))
                if sum(intersected_o2) == 0:
                    raise IndexError
                # should a track intersect 2 o2 regions, load the bigger intersection
                elif sum(intersected_o2) > 1:
                    intersections = o2regions.intersection(
                        shapely.oriented_envelope(buffered_points.geometry.unary_union)
                        ).set_crs(4326).to_crs(3413)
                    o2code = o2regions.loc[intersections.geometry.area.idxmax(), "o2region"]
                else:
                    o2code = o2regions[intersected_o2]["o2region"].values[0]
                o2_extent = load_o2region(o2code).clip_by_rect(*buffered_points.total_bounds)
                if all(o2_extent.is_empty):
                    raise IndexError
                retain_indeces = buffered_points.intersects(o2_extent.unary_union)
                tmp = tmp.isel(time_20_ku=retain_indeces[retain_indeces].index)
            except IndexError:
                warnings.warn("No waveforms left on glacier. Proceeding with empty dataset.")
                tmp = tmp.isel(time_20_ku=[])
        tmp = tmp.assign_attrs(coherence_threshold=coherence_threshold,
                               power_threshold=power_threshold)
        tmp = append_exclude_mask(tmp)
        tmp = append_poca_and_swath_idxs(tmp)
        # ! smooth phase at poca
        
        # add potential phase wrap factor for later use
        tmp = tmp.assign_coords({"phase_wrap_factor": np.arange(-3, 4)})
        super().__init__(data_vars=tmp.data_vars, coords=tmp.coords, attrs=tmp.attrs)
    
    def append_ambiguous_reference_elevation(self):
        # !! This function causes much of the computation time. I suspect that
        # sparse memory accessing can be minimized with some tricks. However,
        # first tries ordering the spatial data, took even (much) longer.
        if not "xph_lats" in self.data_vars:
            self = self.locate_ambiguous_origin()
        # ! tbi: auto download ref dem if not present
        with get_dem_reader() as dem_reader:
            trans_4326_to_dem_crs = Transformer.from_crs('EPSG:4326', dem_reader.crs)
            x, y = trans_4326_to_dem_crs.transform(self.xph_lats, self.xph_lons)
            self = self.assign(xph_x=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), x), 
                               xph_y=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), y))
            with rioxr.open_rasterio(dem_reader) as ref_dem:
                # ! huge improvement potential: instead of the below, rasterio.sample could be used
                # [edit] use postgis
                ref_dem = ref_dem.rio.clip_box(np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)).squeeze()
                self["xph_ref_elevs"] = ref_dem.sel(x=self.xph_x, y=self.xph_y, method="nearest")
        # rasterio suggests sorting like `for ind in np.lexsort([y, x]): rv.append((x[ind], y[ind]))`
        # sort_key = np.lexsort([y, x])
        # planar_coords = zip(x[sort_key], y[sort_key])
        # ref_elev_vector = np.fromiter(dem_reader.sample(planar_coords), "float32")[sort_key.argsort()]
        # return self.assign(xph_ref_elevs=(self.xph_lats.dims,
        #                                   np.reshape(ref_elev_vector,
        #                                              self.xph_lats.shape)))
        return self
    __all__.append("append_ambiguous_reference_elevation")

    def append_best_fit_phase_index(self):
        # ! Implement opt-out or/and grouping alternatives
        if not "group_id" in self.data_vars:
            self = self.tag_groups()
            # it makes sense to always unwrap the phases immediately after finding
            # the groups. assigning the best fitting indices otherwise messes up
            # your data
            self = self.unwrap_phase_diff()
        # before locating echos, find groups because also phase is unwrapped
        if not "xph_elev_diffs" in self.data_vars:
            self = self.append_elev_diff_to_ref()
        self = self.assign(ph_idx=(("time_20_ku", "ns_20_ku"),
                                   np.empty((len(self.time_20_ku), len(self.ns_20_ku)), dtype="int")))
        def min_abs_idx(ph_diff, group_ids):
            out = np.zeros_like(group_ids)
            for i in nan_unique(group_ids):
                mask = group_ids == i
                out[mask] = np.argmin(np.abs(np.sum(ph_diff[mask,:], axis=0)))-3
            return out
        self["ph_idx"] = xr.apply_ufunc(min_abs_idx, 
                                        self.xph_elev_diffs, 
                                        self.group_id,
                                        input_core_dims=[["ns_20_ku", "phase_wrap_factor"], ["ns_20_ku"]],
                                        output_core_dims=[["ns_20_ku"]])
        self["ph_idx"] = xr.where(self.group_id.isnull(), np.abs(self.xph_elev_diffs).idxmin("phase_wrap_factor"), self.ph_idx)
        return self
    __all__.append("append_best_fit_phase_index")
    
    def append_elev_diff_to_ref(self):
        if not "xph_ref_elevs" in self.data_vars:
            self = self.append_ambiguous_reference_elevation()
        self["xph_elev_diffs"] = (self.xph_elevs-self.xph_ref_elevs)
        return self
    __all__.append("append_elev_diff_to_ref")
    
    def append_smoothed_complex_phase(self):
        self["ph_diff_complex_smoothed"] = gauss_filter_DataArray(np.exp(1j*self.ph_diff_waveform_20_ku),
                                                                  dim="ns_20_ku", window_extent=21, std=5)
        return self
    __all__.append("append_smoothed_complex_phase")
    
    @classmethod
    def from_id(cls, track_id: str|pd.Timestamp, **kwargs) -> "l1b_data":
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
    __all__.append("from_id")

    def get_rgi_o2(self) -> str:
        """Finds RGIv7 o2 region that contains the track's central lat,
        lon.

        Returns:
            str: RGI v7 `long_code`
        """
        rgi_o2_gpdf = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
        return rgi_o2_gpdf[rgi_o2_gpdf.contains(
                gpd.points_from_xy(self.lon_20_ku, self.lat_20_ku, crs=4326).unary_union().centroid
            )].long_code.values[0]
    __all__.append("get_rgi_o2")

    def phase_jump(self):
        if not "ph_diff_complex_smoothed" in self.data_vars:
            self = self.append_smoothed_complex_phase()
        ph_diff_diff = self.ph_diff_complex_smoothed.diff("ns_20_ku")
        # ! implement choosing tolerance
        ph_diff_diff_tolerance = .1
        jump_mask =  np.logical_or(np.abs(ph_diff_diff) > ph_diff_diff_tolerance,
                                np.abs(ph_diff_diff).rolling(ns_20_ku=2).sum() > 2* 0.8*ph_diff_diff_tolerance)
        if not "exclude_mask" in self.data_vars:
            self = append_exclude_mask(self)
        return xr.where(self.exclude_mask.sel(ns_20_ku=jump_mask.ns_20_ku), False, jump_mask)
    __all__.append("phase_jump")

    def phase_outlier(self, tol: float|None = None):
        # inputs have to be complex unit vectors
        # if no tol provided calc equivalent of 300 m at nadir
        if tol == None:
            temp_x_width = 300 # [m] allow ph_diff to jump by this value (roughly)
            temp_H = 720e3 # [m] rough altitude of CS2
            # 0s below: set to an arbitrary off nadir angle at which the x_width should actually have the defined value
            tol = (np.arctan(np.tan(np.deg2rad(0))+temp_x_width/temp_H)-np.deg2rad(0)) \
                   * 2*np.pi / np.tan(speed_of_light/Ku_band_freq/antenna_baseline)
        if not "ph_diff_complex_smoothed" in self.data_vars:
            self = self.append_smoothed_complex_phase()
        # ph_diff_tol is small, so approx equal to secant length
        return np.abs(np.exp(1j*self.ph_diff_waveform_20_ku) - self.ph_diff_complex_smoothed) > tol
    __all__.append("phase_outlier")

    # ! rename to something like retrieve_ambiguous_origins
    def locate_ambiguous_origin(self):
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
        # below has been simplified from more transparent:
        # self["xph_elevs"] = r_x - r_N
        # dist_off_groundtrack = r_N/(r_N+self["xph_elevs"]) \
        #                        * r_N*np.arctan(range_to_scat*np.sin(theta)/(r_cs2-range_to_scat*np.cos(theta)))
        dist_off_groundtrack = r_N/r_x  * r_N*np.arctan(range_to_scat*np.sin(theta)/(r_cs2-range_to_scat*np.cos(theta)))
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
    __all__.append("locate_ambiguous_origin")
    
    def ref_range(self):
        corrections = self.mod_dry_tropo_cor_01 \
                      + self.mod_wet_tropo_cor_01 \
                      + self.iono_cor_gim_01 \
                      + self.pole_tide_01 \
                      + self.solid_earth_tide_01 \
                      + self.load_tide_01
        return self.window_del_20_ku/np.timedelta64(1, 's') / 2 * speed_of_light \
               + np.interp(self.time_20_ku, self.time_cor_01, corrections)
    __all__.append("ref_range")
    
    def tag_groups(self):
        phase_outlier = self.phase_outlier()
        ignore_mask = (self.exclude_mask + phase_outlier) != 0
        gap_separator = ignore_mask.rolling(ns_20_ku=3).sum() == 3
        any_separator = np.logical_or(*xr.align(self.phase_jump(), gap_separator, join="outer"))
        rising_edge_per_waveform_counter = (any_separator.astype('int32').diff("ns_20_ku")==-1).cumsum("ns_20_ku") + 1
        group_tags = rising_edge_per_waveform_counter \
            + xr.DataArray(data=np.arange(len(self.time_20_ku))*len(self.ns_20_ku), dims="time_20_ku")
        group_tags = xr.align(group_tags, self.power_waveform_20_ku, join="right")[0].where(~ignore_mask)
        def filter_small_groups(group_ids):
            out = group_ids
            for i in nan_unique(group_ids):
                mask = group_ids == i
                if mask.sum() < 3:
                    out[mask] = 0
            return out
        group_tags = xr.apply_ufunc(filter_small_groups,
                       group_tags,
                       input_core_dims=[["ns_20_ku"]],
                       output_core_dims=[["ns_20_ku"]])
        group_tags = group_tags.where(group_tags != 0)
        self["group_id"] = group_tags
        return self
    __all__.append("tag_groups")

    def to_l2(self, out_vars: list|dict = None, *,
                    retain_vars: list|dict = None,
                    swath_or_poca: str = "swath",
                    **kwargs):
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
            self = self.append_best_fit_phase_index()
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
            buffer = self[out_vars+retain_vars+["ph_idx"]].sel(time_20_ku=waveforms_with_poca)\
                                                          .sel(ns_20_ku=self.poca_idx[~self.poca_idx.isnull()])
            buffer = buffer[out_vars+retain_vars].sel(phase_wrap_factor=buffer.ph_idx).dropna("time_20_ku", how="all")
        elif swath_or_poca == "both":
            swath = self.to_l2(out_vars, retain_vars=retain_vars, swath_or_poca="swath", **kwargs)
            poca = self.to_l2(out_vars, retain_vars=retain_vars, swath_or_poca="poca", **kwargs)
            return swath, poca
        else:
            raise ValueError(f"You provided \"swath_or_poca={swath_or_poca}\". Choose \"swath\", \"poca\",",
                             "or \"both\".")
        drop_coords = [coord for coord in tmp.coords if coord not in ["time", "sample"]]
        from . import l2 # can't be in preamble as this would lead to circularity
        l2_data = l2.from_processed_l1b(tmp.squeeze().drop_vars(drop_coords), **kwargs)
        return l2_data
    __all__.append("to_l2")

    def unwrap_phase_diff(self):
        def unwrap(ph_diff, group_ids):
            out = ph_diff
            for i in nan_unique(group_ids):
                mask = group_ids == i
                out[mask] = np.unwrap(ph_diff[mask])
            return out
        self["ph_diff_waveform_20_ku"] = xr.apply_ufunc(unwrap, 
                                                        self.ph_diff_waveform_20_ku, 
                                                        self.group_id,
                                                        input_core_dims=[["ns_20_ku"], ["ns_20_ku"]],
                                                        output_core_dims=[["ns_20_ku"]])
        return self
    __all__.append("unwrap_phase_diff")

    __all__ = sorted(__all__)

__all__.append("l1b_data")


# helper functions ####################################################
    
def append_exclude_mask(cs_l1b_ds):
    # for now require tuple. could be some auto recognition in future.
    assert(isinstance(cs_l1b_ds.power_threshold, tuple))
    # only signal-to-noise-ratio implemented
    assert(cs_l1b_ds.power_threshold[0] == "snr")
    # if "swath_start" not in cs_l1b_ds:
    #     cs_l1b_ds.append_poca_and_swath_idxs()
    # ! the below is verbose but not fast. scrap the unnecessary computations
    power_threshold = cs_l1b_ds.noise_power_20_ku+10*np.log10(cs_l1b_ds.power_threshold[1])
    cs_l1b_ds["exclude_mask"] = \
        np.logical_or(10*np.log10(cs_l1b_ds.power_waveform_20_ku) < power_threshold,
                      cs_l1b_ds.coherence_waveform_20_ku < cs_l1b_ds.coherence_threshold)
    # if drop_first_peak:
    #     cs_l1b_ds["exclude_mask"] = np.logical_or(cs_l1b_ds.exclude_mask,
    #                                                 (cs_l1b_ds.ns_20_ku < cs_l1b_ds.swath_start).values.T)
    return cs_l1b_ds
__all__.append("append_exclude_mask")
    

def append_poca_and_swath_idxs(cs_l1b_ds):
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


def build_flag_mask(cs_l1b_flag: xr.DataArray, black_list: list = [], white_list: str = "",
                    mode: str = "black_list"):
    if "flag_masks" in cs_l1b_flag.attrs and black_list != [] and mode == "black_list":
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=np.log2(np.abs(cs_l1b_flag.attrs["flag_masks"].astype("int64")
                                                        )).astype("int")).sort_index()
        def flag_func(int_code: int) -> bool:
            for i, b in enumerate(reversed(bin(int_code)[2:])):
                if b == "0": continue
                try:
                    if flag_dictionary.loc[i] in black_list:
                        return True
                except KeyError:
                    raise("Flag not found in attributes! Pointing to a bug or an issue in the data.")
            return False
    # ! below needs a revision: what should be returned?
    elif "flag_values" in cs_l1b_flag.attrs and white_list != "" and mode == "white_list":
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=cs_l1b_flag.attrs["flag_values"])
        def flag_func(int_code: int):
            return flag_dictionary.loc[int_code]
    else:
        raise(NotImplementedError)
    return xr.apply_ufunc(np.vectorize(flag_func), cs_l1b_flag.astype(int), dask="allowed")
__all__.append("build_flag_mask")


def download_files(region_of_interest: str|shapely.Polygon = None,
                   start_datetime: str|pd.Timestamp = "2010",
                   end_datetime: str|pd.Timestamp = "2035", *,
                   buffer_region_by: float = None,
                   track_idx: pd.DatetimeIndex|str,
                   #baseline: str = "latest",
                   ):
    if track_idx is None:
        start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
        track_idx = load_cs_ground_tracks(region_of_interest, start_datetime, end_datetime, buffer_region_by=buffer_region_by).index
    else:
        start_datetime, end_datetime = track_idx.sort_values()[[0,-1]]
    for period in pd.date_range(start_datetime,
                                end_datetime+np.timedelta64(1, 'm'), freq="M"):
        try:
            currently_present_files = os.listdir("../data/L1b/"+period.strftime("%Y/%m"))
        except FileNotFoundError:
            os.makedirs("../data/L1b/"+period.strftime("%Y/%m"))
            currently_present_files = []
        with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
            ftp.login(passwd=personal_email)
            try:
                ftp.cwd("/SIR_SIN_L1/"+period.strftime("%Y/%m"))
            except ftplib.error_perm:
                warnings.warn("Directory /SIR_SIN_L1/"+period.strftime("%Y/%m")+" couldn't be accessed.")
                continue
            else:
                print("\n_______\nentering", period.strftime("%Y - %m"))
            for remote_file in ftp.nlst():
                if remote_file[-3:] == ".nc" \
                and pd.to_datetime(remote_file[19:34]) in track_idx \
                and remote_file not in currently_present_files:
                    local_path = os.path.join("../data/L1b/", period.strftime("%Y/%m"), remote_file)
                    try:
                        with open(local_path, "wb") \
                        as local_file:
                            print("___\ndownloading "+remote_file)
                            ftp.retrbinary("RETR "+remote_file, local_file.write)
                            print("done")
                    except:
                        if os.path.isfile(local_path):
                            os.remove(local_path)
                        raise
__all__.append("download_files")

def download_single_file(track_id: str) -> str:
    # currently only CryoSat-2
    with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
        ftp.login(passwd=personal_email)
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
                        print("___\ndownloading "+remote_file)
                        ftp.retrbinary("RETR "+remote_file, local_file.write)
                        print("done")
                        return local_path
                except:
                    if os.path.isfile(local_path):
                        os.remove(local_path)
                    raise
        print(f"File for id {track_id} couldn't be found in remote dir {ftp.pwd()}.")
        raise FileNotFoundError()
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
