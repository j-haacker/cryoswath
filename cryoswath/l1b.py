import configparser
import fnmatch
import ftplib
import geopandas as gpd
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


config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "config.ini")
if os.path.isfile(config_path):
    config.read(config_path)
try:
    personal_email = config["personal"]["personal_email"]
except:
    print("ESA asks to use one's email as password when downloading data via ftp. Please provide it.")
    response = None
    while response != "y":
        personal_email = input("Your email:")
        response = input(f"Is your email \"{personal_email}\" spelled correctly? (y/n)",).lower()[0]
    if "personal" in config:
        config["personal"]["personal_email"] = personal_email
    else:
        config["personal"] = {"personal_email": personal_email}
    with open(config_path, "w") as config_file:
        config.write(config_file)
    print(f"Thanks. You can change your email in {config_path} manually.")

# requires implicitly rasterio(?), flox(?), dask(?)

class l1b_data(xr.Dataset):
    # for now, only CryoSat-2 implemented
    def __init__(self, l1b_filename: str, *,
                 waveform_selection: list|slice = None,
                 drop_bad_waveforms: bool = True,
                 mask_coherence_gt1: bool = True,
                 drop_non_glacier_areas: bool = True,
                 ) -> None:
        # ! tbi customize or drop misleading attributes of xr.Dataset
        # currently only originally named CryoSat-2 SARIn files implemented
        assert(fnmatch.fnmatch(l1b_filename, "*CS_????_SIR_SIN_1B_*"))
        buffer = xr.open_dataset(l1b_filename)
        # at least until baseline E ns_20_ku needs to be made a coordinate
        buffer = buffer.assign_coords(ns_20_ku=("ns_20_ku", np.arange(len(buffer.ns_20_ku))))
        # first: get azimuth bearing from smoothed incremental azimuths.
        # this needs to be done before dropping part of the recording
        poly3fit_params = np.polyfit(np.arange(len(buffer.time_20_ku)-1), 
                                     WGS84_ellpsoid.inv(lats1=buffer.lat_20_ku[:-1], lons1=buffer.lon_20_ku[:-1],
                                                  lats2=buffer.lat_20_ku[1:], lons2=buffer.lon_20_ku[1:])[0],
                                     3)
        buffer = buffer.assign(azimuth=("time_20_ku", np.poly1d(poly3fit_params)(np.arange(len(buffer.time_20_ku)-.5))%360))
        if waveform_selection != None:
            if not isinstance(waveform_selection, slice) and len(waveform_selection) < 2:
                Exception("You need to select at least 2 waveforms. This is a bug, but not issue filed yet.")
            # assuming index selection
            buffer = buffer.isel(time_20_ku=waveform_selection)
            # print(buffer)
        if mask_coherence_gt1:
            buffer["coherence_waveform_20_ku"] = buffer.coherence_waveform_20_ku.where(buffer.coherence_waveform_20_ku <= 1)
        buffer["power_waveform_20_ku"] = buffer.pwr_waveform_20_ku \
                                       * buffer.echo_scale_factor_20_ku \
                                       * 2**buffer.echo_scale_pwr_20_ku
        if drop_bad_waveforms:
            # see available flags using data.flag.attrs["flag_meanings"]
            buffer = drop_waveform(buffer, build_flag_mask(buffer.flag_mcd_20_ku, [
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
            # ! this takes too long: improve implementation
            # buffer = buffer.isel(time_20_ku=gis.points_on_glacier(gpd.GeoSeries(gpd.points_from_xy(buffer.lon_20_ku, buffer.lat_20_ku), crs=4326)))
            # ! needs to be tidied up:
            # (also: simplify needed?)
            buffered_points = gpd.GeoSeries(gpd.points_from_xy(buffer.lon_20_ku, buffer.lat_20_ku), crs=4326).to_crs(3413).buffer(30_000).simplify(5_000).to_crs(4326)
            o2regions = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
            o2code = o2regions[o2regions.geometry.contains(shapely.box(*buffered_points.total_bounds))]["o2region"].values[0]
            retain_indeces = buffered_points.within(load_o2region(o2code).clip_by_rect(*buffered_points.total_bounds).unary_union)
            if retain_indeces.sum() < 2:
                warnings.warn("Not enough waveforms left on glacier. Proceeding with 2 dummy waveforms to ensure no errors raised.")
                buffer = buffer.isel(time_20_ku=[0,1])
            else:
                print(retain_indeces[retain_indeces].index)
                buffer = buffer.isel(time_20_ku=retain_indeces[retain_indeces].index)
        # add potential phase wrap factor for later use
        buffer = buffer.assign_coords({"phase_wrap_factor": np.arange(-3, 4)})
        super().__init__(data_vars=buffer.data_vars, coords=buffer.coords, attrs=buffer.attrs)
        
    def append_ambiguous_reference_elevation(self):
        # !! This function causes much of the computation time. I
        # suspect that sparse memory accessing can be minimized with
        # some tricks. However, first tries ordering the spatial data,
        # took even (much) longer.
        if not "xph_lats" in self.data_vars:
            self = self.locate_ambiguous_origin()
        # ! tbi: auto download ref dem if not present
        dem_reader = rio.open(os.path.join(dem_path, self.get_rgi_o2()+".tif"), "r")
        trans_4326_to_dem_crs = Transformer.from_crs('EPSG:4326', dem_reader.crs)
        with rioxr.open_rasterio(dem_reader) as ref_dem:
            x, y = trans_4326_to_dem_crs.transform(self.xph_lats, self.xph_lons)
            self = self.assign(xph_x=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), x), 
                               xph_y=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), y))
            # ! huge improvement potential: instead of the below, rasterio.sample could be used
            # [edit] use postgis
            self["xph_ref_elevs"] = ref_dem.sel(x=self.xph_x, y=self.xph_y, method="nearest")
        # rasterio suggests sorting like `for ind in np.lexsort([y, x]): rv.append((x[ind], y[ind]))`
        # sort_key = np.lexsort([y, x])
        # planar_coords = zip(x[sort_key], y[sort_key])
        # ref_elev_vector = np.fromiter(dem_reader.sample(planar_coords), "float32")[sort_key.argsort()]
        # return self.assign(xph_ref_elevs=(self.xph_lats.dims,
        #                                   np.reshape(ref_elev_vector,
        #                                              self.xph_lats.shape)))
        return self

    def append_below_threshold_mask(self, coherence_threshold: float = 0.6, power_threshold: tuple = ("snr", 10)):
        # for now require tuple. could be some auto recognition in future.
        assert(isinstance(power_threshold, tuple))
        # only signal-to-noise-ratio implemented
        assert(power_threshold[0] == "snr")
        power_threshold = self.noise_power_20_ku+10*np.log10(power_threshold[1])
        self["below_thresholds"] = np.logical_or(10*np.log10(self.power_waveform_20_ku) < power_threshold,
                                                self.coherence_waveform_20_ku < coherence_threshold)
        return self

    def append_best_fit_phase_index(self):
        if not "xph_elev_diffs" in self.data_vars:
            self = self.append_elev_diff_to_ref()
        # ! Implement opt-out or/and grouping alternatives
        if not "group_id" in self.data_vars:
            self = self.tag_groups()
        self = self.assign(ph_idx=(("time_20_ku", "ns_20_ku"),
                                   np.empty((len(self.time_20_ku), len(self.ns_20_ku)), dtype="int")))
        self["ph_idx"] = self.xph_elev_diffs.where(self.group_id.isnull()).idxmin("phase_wrap_factor")
        for wf in range(len(self.time_20_ku)):
            if not self.group_id[wf].isnull().all():
                self["ph_idx"][wf][~self.group_id[wf].isnull()] \
                    = self.xph_elev_diffs[wf].groupby(self.group_id[wf]).map(
                        lambda x: x.ns_20_ku*0+x.mean("ns_20_ku").idxmin("phase_wrap_factor"))
        return self
    
    def append_elev_diff_to_ref(self):
        if not "xph_ref_elevs" in self.data_vars:
            self = self.append_ambiguous_reference_elevation()
        self["xph_elev_diffs"] = (self.xph_elevs-self.xph_ref_elevs)
        return self
    
    def append_smoothed_complex_phase(self):
        self["ph_diff_complex_smoothed"] = gauss_filter_DataArray(np.exp(1j*self.ph_diff_waveform_20_ku),
                                                                  dim="ns_20_ku", window_extent=21, std=5)
        return self
    
    @classmethod
    def from_id(cls, track_id: str, **kwargs) -> "l1b_data":
        l1b_data_dir = os.path.join(data_path, "L1b", pd.to_datetime(track_id).strftime(f"%Y{os.path.sep}%m"))
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
                gpd.points_from_xy(self.lon_20_ku, self.lat_20_ku, crs=4326).unary_union().centroid
            )].long_code.values[0]

    def phase_jump(self):
        if not "ph_diff_complex_smoothed" in self.data_vars:
            self = self.append_smoothed_complex_phase()
        ph_diff_diff = self.ph_diff_complex_smoothed.diff("ns_20_ku")
        # ! implement choosing tolerance
        ph_diff_diff_tolerance = .1
        jump_mask =  np.logical_or(np.abs(ph_diff_diff) > ph_diff_diff_tolerance,
                                np.abs(ph_diff_diff).rolling(ns_20_ku=2).sum() > 2* 0.8*ph_diff_diff_tolerance)
        if not "below_threshold" in self.data_vars:
            print("Note: Default thresholds applied.")
            self = self.append_below_threshold_mask()
        return xr.where(self.below_thresholds.sel(ns_20_ku=jump_mask.ns_20_ku), False, jump_mask)

    def phase_outlier(self, tol: float|None = None):
        # inputs have to be complex unit vectors
        # if no tol provided calc equivalent of 300 m at nadir
        if tol == None:
            temp_x_width = 300 # [m] allow ph_diff to jump by this value (roughly)
            temp_H = 720e3 # [m] rough altitude of CS2
            # 0s below: set to an arbitrary off nadir angle at which the x_width should actually have the defined value
            tol = (np.arctan(np.tan(np.deg2rad(0))+temp_x_width/temp_H)-np.deg2rad(0)) \
                   * 2*np.pi / np.tan(speed_of_light/Ku_band_freq/antenna_baseline)
        if not "below_threshold" in self.data_vars:
            print("Note: Default thresholds applied.")
            self = self.append_below_threshold_mask()
        if not "ph_diff_complex_smoothed" in self.data_vars:
            self = self.append_smoothed_complex_phase()
        # ph_diff_tol is small, so approx equal to secant length
        return xr.where(self.below_thresholds, False, np.abs(np.exp(1j*self.ph_diff_waveform_20_ku)
                                                             - self.ph_diff_complex_smoothed) > tol)

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
                           xph_elevs=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), (r_x - r_N).values),
                           xph_thetas=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), theta.values),
                           xph_dists=(("time_20_ku", "ns_20_ku", "phase_wrap_factor"), dist_off_groundtrack.values))
    
    def ref_range(self):
        corrections = self.mod_dry_tropo_cor_01 \
                      + self.mod_wet_tropo_cor_01 \
                      + self.iono_cor_gim_01 \
                      + self.pole_tide_01 \
                      + self.solid_earth_tide_01 \
                      + self.load_tide_01
        return self.window_del_20_ku/np.timedelta64(1, 's') / 2 * speed_of_light \
               + np.interp(self.time_20_ku, self.time_cor_01, corrections)
    
    def tag_groups(self):
        non_group_samples = self.phase_outlier()
        group_tags = (np.logical_or(np.logical_or(self.phase_jump(), non_group_samples.rolling(ns_20_ku=3).sum()==0),
                                    self.below_thresholds).astype('uint8').diff("ns_20_ku")==-1).cumsum("ns_20_ku") \
                     + xr.DataArray(data=np.arange(len(self.time_20_ku))*len(self.ns_20_ku), dims="time_20_ku")
        self["group_id"] = group_tags.where(~self.below_thresholds).where(~non_group_samples)
        return self

    def to_l2(self,
              out_vars: list|dict = {"time_20_ku": "time", "xph_x": "x", "xph_y": "y",
                                     "xph_elevs": "height", "xph_ref_elevs": "h_ref", "xph_elev_diffs": "h_diff"}, 
              retain_vars: list|dict = [], 
              ):
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
        # print(self[out_vars+retain_vars].to_dataframe())
        buffer = self[out_vars+retain_vars].where(~self.below_thresholds)\
                                           .sel(phase_wrap_factor=self.ph_idx)\
                                           .dropna("time_20_ku", how="all")
        # print(buffer.drop_vars(buffer.coords).drop_dims("band"))
        drop_coords = [coord for coord in buffer.coords if coord not in ["time"]]
        from . import l2 # needs to stay here to prevent circular import!
        return l2.from_processed_l1b(buffer.squeeze().drop_vars(drop_coords))#


# helper functions ####################################################
    
def build_flag_mask(cs_l1b_flag: xr.DataArray, black_list: list = [], white_list: str = "",
                    mode: str = "black_list"):
    if "flag_masks" in cs_l1b_flag.attrs and black_list != [] and mode == "black_list":
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=np.log2(np.abs(cs_l1b_flag.attrs["flag_masks"].astype("int64")
                                                        )).astype("int")).sort_index()
        def flag_func(int_code):
            for i, b in enumerate(reversed(bin(int(int_code))[2:])):
                if b == "0": continue
                try:
                    if flag_dictionary.loc[i] in black_list:
                        return True
                except KeyError:
                    raise("Flag not found in attributes! Pointing to a bug or an issue in the data.")
            return False
    elif "flag_values" in cs_l1b_flag.attrs and white_list != "" and mode == "white_list":
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=cs_l1b_flag.attrs["flag_values"])
        def flag_func(int_code):
            return flag_dictionary.loc[int(int_code)]
    else:
        raise(NotImplementedError)
    return xr.apply_ufunc(flag_func, cs_l1b_flag, vectorize=True)

def download_files(region_of_interest: shapely.Polygon, # buffered region in 4326
                    start_datetime: str = "2010-10-10",
                    end_datetime: str = "2011-11-11",
                    baseline: str = "latest"):
    # ! not tested

    # currently only CryoSat-2
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)
    cs_tracks = gis.load_cs_ground_tracks()
    cs_tracks = cs_tracks.loc[start_datetime:end_datetime]
    lon1, lat1, lon2, lat2 = region_of_interest.total_bounds
    # find all tracks that intersect the buffered region of interest.
    # mind that this are calculations on a sphere. currently, the
    # polygon is transformed to ellipsoidial coordinates. not a 100 %
    # sure that this doesn't raise issues close to the poles.
    cs_tracks = cs_tracks.cx[lon1:lon2,lat1:lat2]
    for period in pd.date_range(start_datetime,
                                end_datetime+np.timedelta64(1, 'm'), freq="M"):
        print("\n_______\nentering", period.strftime("%Y - %m"))
        try:
            currently_present_files = os.listdir("../data/L1b/"+period.strftime("%Y/%m"))
        except FileNotFoundError:
            os.makedirs("../data/L1b/"+period.strftime("%Y/%m"))
            currently_present_files = []
        with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
            ftp.login(passwd=personal_email)
            ftp.cwd("/SIR_SIN_L1/"+period.strftime("%Y/%m"))
            for remote_file in ftp.nlst():
                if remote_file[-3:] == ".nc" \
                and pd.to_datetime(remote_file[19:34]) in cs_tracks[cs_tracks.intersects(gis.buffer_4326_shp(region_of_interest.unary_union, 30000))].index \
                and remote_file not in currently_present_files:
                    local_path = os.path.join(["../data/L1b/", period.strftime("%Y/%m"), remote_file])
                    try:
                        with open(local_path, "w+") \
                        as local_file:
                            print("___\ndownloading "+remote_file)
                            ftp.retrbinary("RETR "+remote_file, local_file.write)
                            print("done")
                    except:
                        if os.path.isfile(local_path):
                            os.remove(local_path)
                        raise

def download_single_file(track_id: str) -> str:
    # currently only CryoSat-2
    with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
        ftp.login(passwd=personal_email)
        ftp.cwd("/SIR_SIN_L1/"+pd.to_datetime(track_id).strftime("%Y/%m"))
        for remote_file in ftp.nlst():
            if remote_file[-3:] == ".nc" \
            and remote_file[19:34] == track_id:
                local_path = os.path.join("../data/L1b/", pd.to_datetime(track_id).strftime("%Y/%m"))
                if not os.path.isdir(local_path):
                    os.makedirs(local_path)
                local_path = os.path.join(local_path, remote_file)
                try:
                    with open(local_path, "w+") as local_file:
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

def drop_waveform(cs_l1b_ds, time_20_ku_mask):
    """Use mask along time dim to drop waveforms.

    Args:
        time_20_ku_mask (1-dim bool): Mask: drop where True.

    Returns:
        xr.Dataset or DataArray: Input dataset without marked waveforms.
    """
    return cs_l1b_ds.sel(time_20_ku=cs_l1b_ds.time_20_ku[~time_20_ku_mask])

def choose_group_phase_wrap(waveform):
    # this should be possible for all waveforms in parallel (see below). However, this takes much longer for some
    # reason. Check again, when using dask (looks like a sparse memory accessing issue).
    # ds["ph_idx"][~ds.group_id.isnull()] = ds.xph_swath_h_diff.groupby(ds.group_id).map(lambda x: x.ns_20_ku*0+x.mean("stacked_time_20_ku_ns_20_ku").idxmin("phase_wrap_factor"))
    return waveform.xph_elev_diffs.groupby(waveform.group_id).map(lambda x: x.ns_20_ku*0+x.mean("ns_20_ku").idxmin("phase_wrap_factor"))

def get_buffered_glacier_outlines(l1b_ds):
    # Implement finding polygon based on track lat/lon
    glacier_shp_filename = "../../2022__Novaya_Zemlya/data/shape_files/RGI60_glaciers__Nowaya_Zemlya.shp"
    buffered_glaciered_area_polygon = gpd.read_file(glacier_shp_filename)
    return gpd.GeoSeries(buffered_glaciered_area_polygon.unary_union,
                         crs=buffered_glaciered_area_polygon.crs)\
                             .to_crs(3414).buffer(30000).to_crs(4326)


def get_xtrack_dem_transect(waveform, half_beam_width=15e3, sampling_interval=100):
    sampling_dist = np.arange(-half_beam_width, half_beam_width+1, sampling_interval)
    num_samples = len(sampling_dist)
    lats, lons = WGS84_ellpsoid.fwd(lons=[waveform.lon_20_ku]*num_samples,
                              lats=[waveform.lat_20_ku]*num_samples,
                              az=[waveform.azimuth+90]*num_samples,
                              dist=sampling_dist)[1::-1]
    dem_reader = rio.open(os.path.join(dem_path, "09-02_novaya_zemlya.tif"), "r")
    trans_4326_to_dem_crs = Transformer.from_crs("EPSG:4326", dem_reader.crs)
    xs, ys = trans_4326_to_dem_crs.transform(lats, lons)
    return gpd.GeoDataFrame(dict(elevation=list(dem_reader.sample([(x, y) for x, y in zip(xs, ys)])),
                                 geometry=list(shapely.geometry.Point(x, y) for x, y in zip(xs, ys))),
                            crs=dem_reader.crs)


def get_rot_mat_dem_to_waveform(dem_central_lon, wf_lon, wf_az):
    # Assumes prime meridian to be the same
    return Rotation.from_euler('z', dem_central_lon-wf_lon+wf_az, degrees=True)




    