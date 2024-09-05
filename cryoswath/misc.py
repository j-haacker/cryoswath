import configparser
from dateutil.relativedelta import relativedelta
from defusedxml.ElementTree import fromstring as ET_from_str
import fnmatch
import ftplib
import geopandas as gpd
import glob
import h5py
import inspect
import numpy as np
import os
import pandas as pd
from pyproj import Geod
import queue
import rasterio
import re
from scipy.constants import speed_of_light
import scipy.stats
import shapely
import shutil
from tables import NaturalNameWarning
import time
import threading
from typing import Union
import warnings
import xarray as xr

from . import gis

# make contents accessible
__all__ = ["WGS84_ellpsoid", "antenna_baseline", "Ku_band_freq", "sample_width",     # vars ...........................
           "speed_of_light", "cryosat_id_pattern", "nanoseconds_per_year",
           "cs_id_to_time", "cs_time_to_id", "find_region_id", "flag_translator",    # funcs ..........................
           "gauss_filter_DataArray", "get_dem_reader", "load_cs_full_file_names", 
           "load_cs_ground_tracks", "load_o1region", "load_o2region", "load_basins",
          ]

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
__all__.append("personal_email")


## Paths ##############################################################
data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "data")
l1b_path = os.path.join(data_path, "L1b")
l2_swath_path = os.path.join(data_path, "L2_swath")
l2_poca_path = os.path.join(data_path, "L2_poca")
l3_path = os.path.join(data_path, "L3")
l4_path = os.path.join(data_path, "L4")
tmp_path = os.path.join(data_path, "tmp")
aux_path = os.path.join(data_path, "auxiliary")
cs_ground_tracks_path = os.path.join(aux_path, "CryoSat-2_SARIn_ground_tracks.feather")
rgi_path = os.path.join(aux_path, "RGI")
dem_path = os.path.join(aux_path, "DEM")
__all__.extend(["data_path",
                "l1b_path", "l2_swath_path", "l2_poca_path", "l3_path", "l4_path", "tmp_path",
                "aux_path", "cs_ground_tracks_path", "rgi_path", "dem_path"])

## Config #############################################################
WGS84_ellpsoid = Geod(ellps="WGS84")
# The following is advised to set for pandas<v3 (default for later versions)
pd.options.mode.copy_on_write = True

## Constants ##########################################################
antenna_baseline = 1.1676
Ku_band_freq = 13.575e9
sample_width = speed_of_light/(320e6*2)/2
cryosat_id_pattern = re.compile("20[12][0-9][01][0-9][0-3][0-9]T[0-2][0-9]([0-5][0-9]){2}")
nanoseconds_per_year = 365.25*24*60*60*1e9

## Functions ##########################################################

# security issue?
class binary_chache():
    __all__ = []
    def __init__(self):
        self._cache = bytearray()
    
    @property
    def cache(self):
        return self._cache.decode()
    __all__.append("cache")

    @cache.deleter
    def cache(self):
        del self._cache[:]

    def add(self, new_part):
        self._cache.extend(new_part)
    __all__.append("add")
__all__.append("binary_chache")


def cs_id_to_time(cs_id: str) -> pd.Timestamp:
    return pd.to_datetime(cs_id, format="%Y%m%dT%H%M%S")


def cs_time_to_id(time: pd.Timestamp) -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def convert_all_esri_to_feather(dir_path: str = None) -> None:
    for shp_file in glob.glob("*.shp", root_dir=dir_path):
        try:
            gis.esri_to_feather(os.path.join(dir_path, shp_file))
        except Exception as err:
            print("Error occured while translating", shp_file, " ... skipped.")
            print("Error message:", str(err))
        else:
            print("Converted", shp_file)
            basename = os.path.extsep.join(shp_file.split(os.path.extsep)[:-1])
            for associated_file in glob.glob(basename+".*", root_dir=dir_path):
                if associated_file.split(os.path.extsep)[-1] != "feather":
                    try:
                        os.remove(os.path.join(dir_path, associated_file))
                    except Exception as err:
                        print("Couldn't clean up", associated_file, " ... skipped.")
                        print("Error message:", str(err))
                    else:
                        print("Removed", associated_file)
__all__.append("convert_all_esri_to_feather")


# def download_file(url: str, out_path: str = ".") -> str:
#     # snippet adapted from https://stackoverflow.com/a/16696317
#     # authors: https://stackoverflow.com/users/427457/roman-podlinov
#     #      and https://stackoverflow.com/users/12641442/jenia
#     local_filename = os.join(out_path, url.split('/')[-1])
#     # NOTE the stream=True parameter below
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(local_filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192): 
#                 # If you have chunk encoded response uncomment if
#                 # and set chunk_size parameter to None.
#                 #if chunk: 
#                 f.write(chunk)
#     return local_filename
# __all__.append("download_file")
# not used and clutters namespace
# def download_file(url: str, out_path: str = ".") -> str:
#     # snippet adapted from https://stackoverflow.com/a/16696317
#     # authors: https://stackoverflow.com/users/427457/roman-podlinov
#     #      and https://stackoverflow.com/users/12641442/jenia
#     local_filename = os.join(out_path, url.split('/')[-1])
#     # NOTE the stream=True parameter below
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(local_filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192): 
#                 # If you have chunk encoded response uncomment if
#                 # and set chunk_size parameter to None.
#                 #if chunk: 
#                 f.write(chunk)
#     return local_filename


def extend_filename(file_name: str, extension: str) -> str:
    fn_parts = file_name.split(os.path.extsep)
    return os.path.extsep.join(fn_parts[:-1]) + extension + os.path.extsep + fn_parts[-1]
__all__.append("extend_filename")


# ! make recursive
def filter_kwargs(func: callable,
                  kwargs: dict, *,
                  blacklist: list[str] = None,
                  whitelist: list[str] = None,
                  ) -> dict:
    def ensure_list(tmp_list):
        if tmp_list is None: return []
        elif isinstance(tmp_list, str): return [tmp_list]
        else: return tmp_list
    blacklist = ensure_list(blacklist)
    whitelist = ensure_list(whitelist)
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if (k in params and k not in blacklist) or k in whitelist}
__all__.append("filter_kwargs")


def find_region_id(location: any, scope: str = "o2") -> str:
    if isinstance(location, xr.DataArray) or isinstance(location, xr.Dataset):
        left, lower, right, upper = location.rio.transform_bounds(4326)
        location = shapely.Point(left+(right-left)/2, lower+(upper-lower)/2)
    if isinstance(location, gpd.GeoDataFrame):
        location = location.geometry
    if isinstance(location, gpd.GeoSeries):
        location = location.to_crs(4326).union_all("coverage")
    if not isinstance(location, shapely.Geometry):
        if isinstance(location, tuple) or (isinstance(location, list) and len(location)<3):
            location = shapely.Point(location[1], location[0])
        else:
            location = shapely.Polygon([(coord[1], coord[0]) for coord in location])
    rgi_o2_gpdf = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
    rgi_region = rgi_o2_gpdf[rgi_o2_gpdf.contains(location.centroid)]
    if scope == "o1":
        return rgi_region["o1region"].values[0]
    elif scope == "o2":
        out = rgi_region["o2region"].values[0]
        if out == "05-01":
            sub_o2 = gpd.GeoSeries([load_o2region(f"05-1{i+1}").union_all("coverage").envelope for i in range(5)], crs=4326)
            contains_location = sub_o2.contains(location)
            if not any(contains_location):
                raise Exception(f"Location {location} not in any of Greenlands subregions (N,W,SW,SE,E).")
            elif sum(contains_location) > 1:
                raise Exception(f"Location {location} is in multiple subregions (N,W,SW,SE,E).")
            out = f"05-1{int(sub_o2[contains_location].index.values)+1}"
        return out
    elif scope == "basin":
        rgi_glacier_gpdf = load_o2region(rgi_region["o2region"].values[0], "glaciers")
        return rgi_glacier_gpdf[rgi_glacier_gpdf.contains(location.centroid)]["rgi_id"].values[0]
    raise Exception("`scope` can be one of \"o1\", \"o2\", or \"basin\".")

    # ! tbi: if only small region/one glacier, make get its
    # to_planar = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3413))
    # if shapely.ops.transform(to_planar.transform, region_outlines).area > 


def flag_outliers(data, *,
                  weights = None,
                  stat: callable = np.median,
                  deviation_factor: float = 3,
                  scaling_factor: float = 2*2**.5*scipy.special.erfinv(.5)):
    """Flags data that is considered outlier given a set of assumptions.

    Data too far from a reference point is marked. Works analogous comparing
    data to its mean in terms of standard deviations.

    Function was meant to be versatile. However, I'm not sure it makes
    sense using it with other than the "usual" statistics: mean and median.

    It defaults to marking data further from the median than 3 scaled MADs.

    Args:
        data (ArrayLike): If data is an array, outliers will be flagged
            along first dimension (given `stat` works like most numpy
            functions).
        weights (ArrayLike): If weights are provided, they are passed as
            the keyword argument to `stat`.
        stat (callable, optional): Function to return first and second
            reference points. Defaults to np.median.
        deviation_factor (float, optional): Allowed number of reference
            point distances between data and first reference point.
            Defaults to 3.
        scaling_factor (float, optional): Reference distance scaling.
            Defaults to 2*2**.5*scipy.special.erfinv(.5)).

    Returns:
        bool, shaped like input: Mask that is positive for outliers.
    """
    if weights is None:
        first_moment = stat(data)
    else:
        first_moment = stat(data, weights=weights)
    # print(first_moment)
    deviation = np.abs(data - first_moment)
    # print(deviation)
    if weights is None:
        deviation_limit = stat(deviation) * deviation_factor * scaling_factor
    else:
        deviation_limit = stat(deviation, weights=weights) * deviation_factor * scaling_factor
    # print(deviation_limit)
    return deviation > deviation_limit
__all__.append("flag_outliers")


def flag_translator(cs_l1b_flag):
    """Retrieves the meaning of a flag from the attributes.
    
    If attributes contain "flag_masks", it converts the value to a
    binary mask and returns a list of flags. Else it expects
    "flag_values" and interprets and returns the flag as one of a set of
    options.

    This works for CryoSat-2 L1b netCDF data. It depends on the
    attribute structure and names.

    Args:
        cs_l1b_flag (0-dim xarray.DataArray): Flag variable of waveform.

    Returns:
        list or string: List of flags or single option, depending on
        flag.
    """
    if "flag_masks" in cs_l1b_flag.attrs:
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=np.log2(np.abs(cs_l1b_flag.attrs["flag_masks"].astype("int64")
                                                        )).astype("int")).sort_index()
        bin_str = bin(int(cs_l1b_flag.values))[2:]
        flag_list = []
        for i, b in enumerate(reversed(bin_str)):
            if b == "1":
                try:
                    flag_list.append(flag_dictionary.loc[i])
                except KeyError:
                    raise(f"Unkown flag: {2**i}! This points to a bug either in the code or in the data!")
        return flag_list
    else:
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=cs_l1b_flag.attrs["flag_values"])
        return flag_dictionary.loc[int(cs_l1b_flag.values)]


def gauss_filter_DataArray(da, dim, window_extent, std):
    # force window_extent to be uneven to ensure center to be where expected
    half_window_extent = window_extent//2
    window_extent = 2*half_window_extent+1
    gauss_weights = scipy.stats.norm.pdf(np.arange(-half_window_extent, half_window_extent+1), scale=std)
    gauss_weights = xr.DataArray(gauss_weights/np.sum(gauss_weights), dims=["window_dim"])
    if np.iscomplexobj(da):
        helper = da.rolling({dim: window_extent}, center=True, min_periods=1).construct("window_dim").dot(gauss_weights)
        return helper/np.abs(helper)
    else:
        return da.rolling({dim: window_extent}, center=True, min_periods=1).construct("window_dim").dot(gauss_weights)


def get_dem_reader(data: any = None) -> rasterio.DatasetReader:
    if "lat_20_ku" in data:
        lat = data.lat_20_ku.values[0]
    elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        lat = np.mean(data.rio.transform_bounds("EPSG:4326")[1::2])
    elif isinstance(data, gpd.GeoSeries) or isinstance(data, gpd.GeoDataFrame):
        lat = data.to_crs(4326).centroid.y
    elif isinstance(data, shapely.Geometry):
        lat = np.mean(data.total_bounds[[1,3]])
    elif isinstance(data, float) \
            or isinstance(data, int) \
            or (isinstance(data, np.ndarray) and data.size==1):
        lat = data
    elif isinstance(data, str):
        if data.lower() in ["arctic", "arcticdem"]:
            lat = 90
        elif data.lower() in ["antarctic", "rema"]:
            lat = -90
    if "lat" not in locals():
        raise NotImplementedError(f"`get_dem_reader` could not handle the input of type {data.__class__}. See doc for further info.")
    if lat > 0:
        return rasterio.open(os.path.join(dem_path, "arcticdem_mosaic_100m_v4.1_dem.tif"))
    else:
        return rasterio.open(os.path.join(dem_path, "rema_mosaic_100m_v2.0_filled_cop30_dem.tif"))


def interpolate_hypsometrically(ds: xr.Dataset,
                                main_var: str,
                                elev: str = "ref_elev",
                                weights: str = "weights",
                                degree: int = 3) -> xr.Dataset:
    if "time" in ds.dims:
        return ds.groupby("time").apply(interpolate_hypsometrically, main_var=main_var, elev=elev, weights=weights, degree=degree)
    ds[weights] = xr.where(ds[elev].isnull(), np.nan, ds[weights])
    # abort if too little data (checking elevation and data validity).
    # necessary to prevent errors but also introduces data gaps
    if ds[elev].where(ds[weights]>0).count() < degree*3:
        print("too little data")
        return ds
    # also, abort if there isn't anything to do
    if not (ds[weights]==0).any():
        print("nothing to do")
        return ds
    ds[weights] = ds[weights]/ds[weights].where(ds[weights]>0).mean()
    # first fit
    x0 = ds[elev].where(ds[weights]>0).mean().values
    tmp = ds.to_dataframe()[[main_var, elev, weights]].dropna(axis=0, how="any")
    coeffs = np.polyfit(tmp[elev]-x0, tmp[main_var], degree, w=tmp[weights])
    residuals = np.polyval(coeffs, tmp[elev]-x0) - tmp[main_var]
    # find and remove outlier
    outlier_mask = flag_outliers(residuals[tmp[weights]>0], deviation_factor=5)

    # # debugging tool
    # import matplotlib.pyplot as plt
    # # print(df_only_valid.ref_elev[weights>0], df_only_valid._median[weights>0], 1/weights[weights>0], '_')
    # plt.errorbar(df_only_valid.ref_elev[weights>0], df_only_valid._median[weights>0], yerr=.1/weights[weights>0], fmt='_')
    # plt.plot(df_only_valid.ref_elev[weights>0][outlier_mask], df_only_valid._median[weights>0][outlier_mask], 'rx')
    # pl_x = np.arange(0, 2001, 100)
    # plt.plot(pl_x, np.polyval(coeffs, pl_x), '-')
    # plt.show()

    tmp.loc[tmp[weights]>0,weights] = ~outlier_mask.values * tmp.loc[tmp[weights]>0,weights]
    # fit again
    coeffs = np.polyfit(tmp[elev]-x0, tmp[main_var], degree, w=tmp[weights])
    lowest = tmp.loc[tmp[weights]>0,elev].min()
    highest = tmp.loc[tmp[weights]>0,elev].max()
    ds[main_var] = xr.where(ds[weights]==0,
                            xr.where(ds[elev]>lowest,
                                     xr.where(ds[elev]<highest,
                                              np.polyval(coeffs, ds[elev]-x0),
                                              np.polyval(coeffs, highest-x0)), 
                                     np.polyval(coeffs, lowest-x0)),
                            ds[main_var])
    # set weights to nan, so that the grid cells do not get overwritten in a
    # second interpolation cycle
    ds[weights] = xr.where(ds[weights]==0, np.nan, ds[weights])
    return ds
__all__.append("interpolate_hypsometrically")


def load_cs_full_file_names(update: str = "no") -> pd.Series:
    """Loads a pandas.Series of the original CryoSat-2 L1b file names.

    Having the file names available can be handy to organize your local 
    data.

    This function can be used to update your local list by setting `update`.

    Args:
        update (str, optional): One of "no", "quick", "regular, or "full".
            "quick" continues from the last locally known file name,
            "regular" checks for changes between the stages OFFL and LTA,
            and "full" replaces the local data base with a new one. Defaults
            to "no".

    Returns:
        pd.Series: Full L1b file names without path or extension.
    """
    file_names_path = os.path.join(aux_path, "CryoSat-2_SARIn_file_names.pkl")
    if os.path.isfile(file_names_path):
        file_names = pd.read_pickle(file_names_path).sort_index()
    if update == "no":
        return file_names
    if update == "quick":
        last_lta_idx = file_names.index[-1]
        print(last_lta_idx+pd.offsets.MonthBegin(-1, normalize=True))
    if update == "regular":
        # ! "regular" should also be baseline and version aware
        last_lta_idx = file_names[(fn[3:7]=="LTA_" for fn in file_names)].index[-1]
        print(last_lta_idx+pd.offsets.MonthBegin(-1, normalize=True))
    with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
        ftp.login(passwd=personal_email)
        ftp.cwd("/SIR_SIN_L1")
        for year in ftp.nlst():
            if update != "full" and year < str(last_lta_idx.year):
                print("skip", year)
                continue
            try:
                ftp.cwd(f"/SIR_SIN_L1/{year}")
                print(f"entered /SIR_SIN_L1/{year}")
                for month in ftp.nlst():
                    if update != "full" and pd.to_datetime(f"{year}-{month}") \
                                            < last_lta_idx+pd.offsets.MonthBegin(-1, normalize=True):
                        print("skip", month)
                        continue
                    print(f"cwd /SIR_SIN_L1/{year}/{month}")
                    ftp.cwd(f"/SIR_SIN_L1/{year}/{month}")
                    print(f"scanning /SIR_SIN_L1/{year}/{month}")
                    for remote_file in ftp.nlst():
                        if remote_file[-3:] == ".nc":
                            remote_idx = pd.to_datetime(remote_file[19:34])
                            if update == "regular" and remote_idx in file_names.index \
                                    and (file_names.loc[remote_idx][3:7]=="LTA_" or remote_file[3:7]=="OFFL"):
                                continue
                            file_names.loc[remote_idx] = remote_file[:-3]
            except Exception:
                warnings.warn(f"Error occurred in remote directory /SIR_SIN_L1/{year}/{month}.")
            finally:
                file_names.to_pickle(file_names_path)
                return file_names


def load_cs_ground_tracks(region_of_interest: str|shapely.Polygon = None,
                          start_datetime: str|pd.Timestamp = "2010",
                          end_datetime: str|pd.Timestamp = "2030", *,
                          buffer_period_by: relativedelta = None,
                          buffer_region_by: float = None,
                          update: str = "no",
                          n_threads: int = 8,
                          ) -> gpd.GeoDataFrame:
    """Read the GeoDataFrame of CryoSat-2 tracks from disk.

    If desired, you can query certain extents or periods by specifying
    arguments.

    Further, you can update the database by setting `update` to "regular" or
    "full". Mind that this typically takes some time (regular on the order
    of minutes, full rather hours).

    Args:
        region_of_interest (str | shapely.Polygon, optional): Can be any RGI
            code or a polygon in lat/lon (CRS EPSG:4326). If requesting o1
            regions, provide the long code, e.g., "01_alaska". Defaults to None.  
        start_datetime (str | pd.Timestamp, optional): Defaults to "2010".  
        end_datetime (str | pd.Timestamp, optional): Defaults to "2030".  
        buffer_period_by (relativedelta, optional): Extends the period to
            both sides. Handy if you use this function to query tracks for an
            aggregated product. Defaults to None.  
        buffer_region_by (float, optional): Handy to also query tracks in the
            proximity that may return elevation estimates for your region of
            interest. Unit are meters here. CryoSat's footprint is +- 7.5 km to
            both sides, anything above 30_000 does not make much sense. Defaults
            to None.  
        update (str, optional): If you are interested in the latest tracks,
            update frequently with `update="regular"`. If you believe tracks are
            missing for some reason, choose `update="full"` (be aware this takes
            a while). Defaults to "no".  
        n_threads (int, optional): Number of parallel ftp connections. If you
            choose too many, ESA will refuse the connection. Defaults to 8.

    Raises:
        ValueError: For invalid `update` arguments.

    Returns:
        gpd.GeoDataFrame: CryoSat-2 tracks.
    """
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    if os.path.isfile(cs_ground_tracks_path):
        cs_tracks = gpd.read_feather(cs_ground_tracks_path)
        if "index" in cs_tracks.columns: 
            cs_tracks.set_index("index", inplace=True)
        cs_tracks.index = pd.to_datetime(cs_tracks.index)
        cs_tracks.sort_index(inplace=True)
    else:
        cs_tracks = gpd.GeoSeries()
        update = "full"
    if update == "full":
        last_idx = pd.Timestamp("2010-07-01")
    # ! should be consistent with load names -> rather call it "quick"?
    elif update == "regular":
        last_idx = pd.to_datetime(cs_tracks.index[-1])
    elif update != "no":
        raise ValueError("Allowed values for `update` are \"full\". \"regular\". or \"no\". "
                         +f"You set it to \"{update}\".")
    if update != "no":

        # the next two function have only a local purpose.
        def save_current_track_list(new_track_series: gpd.GeoSeries):
            """saves the tracklist; backing up the old if older than 5 days."""
            if not os.path.isfile(extend_filename(cs_ground_tracks_path, "__backup"))\
                    or time.time() - os.path.getmtime(cs_ground_tracks_path)  > 5 * 24*60*60:
                print("backing up \"old\" track file")
                shutil.copyfile(cs_ground_tracks_path, extend_filename(cs_ground_tracks_path, "__backup"))
            print("saving current track list to file")
            new_track_series.to_feather(cs_ground_tracks_path)

        def collect_missing_tracks(remote_files: list[str],
                                   present_tracks: gpd.GeoSeries) -> gpd.GeoSeries:
            """Gets track if not in list already.

            Args:
                files (list[str]): HDR file names. All of the same month.
                present_tracks (gpd.GeoSeries): Known tracks.

            Returns:
                gpd.GeoSeries: Missing tracks to be added to the collection.
            """
            with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
                ftp.login(passwd=personal_email)
                ftp.cwd("/SIR_SIN_L1/"+pd.to_datetime(remote_files[0][19:34]).strftime("%Y/%m"))
                tracks_to_be_added = gpd.GeoDataFrame(columns=["geometry"]).rename_axis("index")
                for rf_name in remote_files:
                    if fnmatch.fnmatch(rf_name, "CS_????_SIR_SIN_1B_*.HDR"):
                        if pd.to_datetime(rf_name[19:34]) in present_tracks.index:
                            continue
                        cache = binary_chache()
                        ftp.retrbinary('RETR '+rf_name, cache.add)
                        et = ET_from_str(cache.cache)
                        root = et.find("Variable_Header/SPH/Product_Location")
                        coordinates = {coord: int(root.find(coord).text)/1e6 for coord in ["Start_Long", "Start_Lat", "Stop_Long", "Stop_Lat"]}
                        tracks_to_be_added.loc[pd.to_datetime(rf_name[19:34])] = shapely.LineString((
                            [coordinates["Start_Long"], coordinates["Start_Lat"]],
                            [coordinates["Stop_Long"], coordinates["Stop_Lat"]]))
                        if not all([(v > -180) and (v < 360) for k, v in coordinates.items()]):
                            warnings.warn(f"whats with {rf_name} giving {coordinates}?")
                            print("track is:", tracks_to_be_added.loc[rf_name[19:34]])
                    elif rf_name[-3:].lower() != ".nc":
                        warnings.warn("Encountered unexpected file:"+rf_name
                                      +"\n\tShould this appear more often, adapt this function.")
            return tracks_to_be_added

        result_queue = queue.SimpleQueue()
        task_queue = request_workers(collect_missing_tracks, n_threads, result_queue)
        # for each month after last_idx, list all HDR-files and check whether
        # they are in the local collection.
        while True:
            with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
                ftp.login(passwd=personal_email)
                try:
                    ftp.cwd("/SIR_SIN_L1/"+last_idx.strftime("%Y/%m"))
                except ftplib.error_perm:
                    print("couldn't switch to month(?)", last_idx.strftime("%Y/%m"),
                          "This should only concern you, if you do expect tracks there.")
                    break
                remote_files = ftp.nlst()
            # cut the file list into chunks and dispatch to workers
            batch_size = len(remote_files)//(n_threads*3)+1
            while remote_files:
                try:
                    task_queue.put((remote_files[:batch_size], cs_tracks))
                    remote_files[:batch_size] = []
                except IndexError:
                    task_queue.put((remote_files[:], cs_tracks))
                    remote_files[:] = []
            # wait for and collect new tracks
            new_tracks_collection = []
            while not task_queue.empty() or not result_queue.empty():
                try:
                    tmp = result_queue.get(block=True, timeout=10*60)
                    if not tmp.empty:
                        new_tracks_collection.append(tmp)
                except queue.Empty:
                    print("waiting for task queue")
                    time.sleep(10)
            # append to local collection and save the result, if any
            if new_tracks_collection:
                cs_tracks = pd.concat([cs_tracks, pd.concat(new_tracks_collection)], sort=True)
                duplicate = cs_tracks.index.duplicated(keep="last")
                if duplicate.sum() > 0:
                    warnings.warn(f"{duplicate.sum()} duplicates found; dropping them.")
                    cs_tracks = cs_tracks[~duplicate]
                    cs_tracks.sort_index(inplace=True)
                save_current_track_list(cs_tracks)
            print(f"scanned all files in", last_idx.strftime("%Y/%m"))
            last_idx = last_idx + pd.DateOffset(months=1)
            print(f"switching to", last_idx.strftime("%Y/%m"))

    # the local collection has been updated. now, return the tracks
    if buffer_period_by is not None:
        start_datetime = start_datetime - buffer_period_by
        end_datetime = end_datetime + buffer_period_by
    cs_tracks = cs_tracks.loc[start_datetime:end_datetime]
    if region_of_interest is not None:
        if isinstance(region_of_interest, str):
            # union=False neccessary for Greenland and large regions
            region_of_interest = load_glacier_outlines(region_of_interest, union=False).geometry.values
        if buffer_region_by is not None:
            region_of_interest = gis.buffer_4326_shp(region_of_interest, buffer_region_by)
        else:
            region_of_interest = gis.simplify_4326_shp(shapely.ops.unary_union(region_of_interest))
        # find all tracks that intersect the buffered region of interest.
        # mind that this are calculations on a sphere. currently, the
        # polygon is transformed to ellipsoidal coordinates. not a 100 %
        # sure that this doesn't raise issues close to the poles.
        cs_tracks = cs_tracks[cs_tracks.intersects(region_of_interest)]
    return cs_tracks.set_crs(4326)


def load_o1region(o1code: str, product: str = "complexes") -> gpd.GeoDataFrame:
    """Loads RGI v7 basin or complex outlines and meta data.

    Args:
        o1code (str): starting with "01".."20"
        product (str, optional): Either "glaciers" or "complexes". Defaults to "complexes".

    Raises:
        ValueError: If o1code can't be recognized.
        FileNotFoundError: If RGI data is missing.

    Returns:
        gpd.GeoDataFrame: Queried RGI data with geometry column containing the outlines.
    """
    if product == "complexes":
        product = "C"
    elif product in ["glaciers", "basins"]:
        product = "G"
    else:
        raise ValueError(f"Argument product should be either glaciers or complexes not \"{product}\".")
    rgi_files = os.listdir(rgi_path)
    for file in rgi_files:
        if re.match(f"RGI2000-v7\\.0-{product}-{o1code[:2]}_.*", file):
            file_path = os.path.join(rgi_path, file)
            if file.endswith(".feather"):
                # print("reading feather")
                o1region = gpd.read_feather(file_path)
                # print("file read")
            elif file.endswith(".shp") or os.path.isdir(file_path):
                o1region = gpd.read_file(file_path)
            else:
                continue
            break
    if "o1region" not in locals():
        print(f"RGI file RGI2000-v7.0-{product}-{o1code[:2]}_... couldn't be found.",
              "Make sure RGI files are available in data/auxiliary/RGI. If you did",
              "not download them already, you can find them at",
              f"https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-{product}/.",
              "Mind that you need to unzip them. If you decide to put them into a",
              "directory, name it as the file is named (e.g. RGI2000-v7.0-G-01_alaska).")
        raise FileNotFoundError
    if product == "C":
        # ! work-around: drop small glaciers
        # issue: takes long to do computations or kernel crashes if
        # (assumption) region contains too many small glaciers. this is
        # equally true for o2 regions, which is why I drop them here
        # already. observed for the Alps.
        small_glacier_mask = o1region.area_km2 < 1
        if sum(small_glacier_mask) != 0:
            warnings.warn(f"Dropping {sum(small_glacier_mask)} glaciers < 1 km² from RGI o1 region.")
        o1region = o1region[~small_glacier_mask]
    return o1region


def load_o2region(o2code: str, product: str = "complexes") -> gpd.GeoDataFrame:
    o1region = load_o1region(o2code[:2], product)
    # special handling for greenland periphery
    if o2code.startswith("05") and not o2code.endswith("01"):
        return o1region[o1region.intersects(gpd.read_feather(os.path.join(rgi_path, o2code+".feather")).union_all("coverage"))]
    return o1region[o1region["o2region"]==o2code[:5]]


def load_basins(rgi_ids: list[str]) -> gpd.GeoDataFrame:
    if len(rgi_ids) > 1:
        assert(all([id[:17]==rgi_ids[0][:17]] for id in rgi_ids))
    rgi_o1_gpdf = load_o1region(rgi_ids[0].split("-")[3], product="glaciers")
    id_to_index_series = pd.Series(data=rgi_o1_gpdf.index, index=rgi_o1_gpdf.rgi_id)
    return rgi_o1_gpdf.loc[id_to_index_series.loc[rgi_ids].values]


def load_glacier_outlines(identifier: str|list[str],
                          product: str = "complexes",
                          union: bool = True,
                          crs: int|CRS = None,
                          ) -> shapely.MultiPolygon:
    if isinstance(identifier, list):
        out = load_basins(identifier)
    elif len(identifier) == (7+4+1+2+5+4) and identifier.split("-")[:3] == ["RGI2000", "v4.1", "G"]:
        out = load_basins([identifier])
    # the pattern is rather allowing, set it to "^(-?[012][0-9]){2}(_[a-z]+){1,5}(_[0-9][a-z][0-9]?)?$" to make it tight
    elif len(identifier) >= 5 and re.match("^(-?[0-3][0-9]){2}$", identifier[:5]):
        out = load_o2region(identifier[:5], product=product)
    elif re.match("[012][0-9](_[a-z]+)?", identifier):
        out = load_o1region(identifier[:2], product=product)
    else:
        raise ValueError(f"Provided o1, o2, or RGI identifiers. \"{identifier}\" not understood.")
    if crs is not None:
        out = out.to_crs(crs)
    if union: # former default
        try:
            out = out.union_all(method="coverage")
        except:
            out = out.union_all(method="unary")
    return out
__all__.append("load_glacier_outlines")


def merge_l2_cache(source_glob: str,
                   destination_file_name: str,
                   exclude_endswith: list[str] = ["backup", "collection"],
                   ) -> None:
    """Append cached l2 data from various hdf files into one.

    Tests whether data is present in destination; if not, copies the data.

    This function is very specifically for cached l2 data as created by
    `l3.build_dataset`. 

    Args:
        source_glob (str): Unix-like glob pattern to match source files
            in `misc.tmp_path` (default: data/tmp/).
        destination_file_name (str): ... in `misc.tmp_path`.
        exclude_endswith (list[str], optional): Do not include files with
            the specified ending. Useful to exclude backups. Defaults to
            ["backup", "collection"].
    """
    # this snippet turned out useful: one can split the caching process,
    # e.g., into years and combine the cache files using this function
    # afterward.
    # not tested after migrating here from notebook
    with h5py.File(os.path.join(tmp_path, destination_file_name), "a") as h5_dest:
        for source_path in sorted(glob.glob(os.path.join(tmp_path, source_glob))):
            print("\n", source_path)
            if any([source_path.endswith(ending) for ending in exclude_endswith]):
                continue
            with h5py.File(source_path, "r") as h5_src:
                def collect_groups(name, node):
                    if name.split("/")[-1].startswith("t_"):
                        if name not in h5_dest:
                            print(name, "will be copied ...")
                            h5_src.copy(h5_src["/"+name], h5_dest, "/"+name)
                        else:
                            print(name, "exists in collection")
                    else:
                        pass
                        # print(name, "is not an end node")
                h5_src.visititems(collect_groups)
__all__.append("merge_l2_cache")


def nan_unique(data: np.typing.ArrayLike) -> list:
    return [element for element in np.unique(data) if not np.isnan(element)]
__all__.append("nan_unique")


def request_workers(task_func: callable, n_workers: int, result_queue: queue.Queue = None) -> queue.Queue:
    task_queue = queue.Queue()
    def worker():
        while True:
            try:
                next_task = task_queue.get()
            except TypeError:
                continue
            if next_task is not None:
                result = task_func(*next_task)
                if result_queue is not None:
                    result_queue.put(result)
                task_queue.task_done()
    for i in range(n_workers):
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
    return task_queue
__all__.append("request_workers")


def repair_l2_cache(filepath: str, *, region_of_interest: shapely.MultiPolygon = None, force: bool = False) -> None:
    """Attempts to repair corrupted l2 cache files.

    The caching logic is not 100% safe. To repair a cache, this function
    removes duplicates and sorts the data index.
    If the note names for some reason 

    Args:
        filepath (str): Path to l2 cache file.
        region_of_interest (shapely.Geometry, optional): EPSG:4326
            outline of considered region. If provided, removes chunks
            with no points inside projected bounding box of outline.
        force (bool): Disregard file size safety, e.g., if you
            expect less than 2/3 of the data to remain.
    """
    if region_of_interest is not None:
        crs = gis.find_planar_crs(shp=region_of_interest)
        bbox = shapely.box(*gpd.GeoSeries(region_of_interest, crs=4326).to_crs(crs).bounds.values[0])
    def move_node(name, node):
        if isinstance(node, h5py.Dataset):
            pass
        elif "_i_table" in node:
            tmp = pd.read_hdf(tmp_h5, key=node.name)
            if "bbox" not in locals() or any(shapely.within(shapely.points(tmp.x, tmp.y), bbox)):
                tmp.drop_duplicates(keep="first").sort_index().to_hdf(filepath, key=node.name, format="table")
    tmp_h5 = os.path.join(data_path, "tmp", "tmp")
    if os.path.exists(tmp_h5):
        if os.path.isfile(tmp_h5):
            os.remove(tmp_h5)
        elif os.path.isdir(tmp_h5):
            shutil.rmtree(tmp_h5)
        else:
            raise Exception(f"Can't remove {tmp_h5}; neither file nor directory!?")
    # I expect `shutil.move` to be safe and believe: either it succeeds or nothing happens
    shutil.move(filepath, tmp_h5)
    try:
        print("Starting to repair file. This may take several minutes. You can",
             f"monitor the progress by viewing the file sizes of {filepath} and",
             f"{tmp_h5}. They will be similar at the end of the process. It should",
              "be reasonably safe to abort.")
        # below hides warnings about a minus sign in node names. this can safely be ignored.
        warnings.filterwarnings('ignore', category=NaturalNameWarning)
        with h5py.File(tmp_h5, "r") as h5:
            h5.visititems(move_node)
        warnings.filterwarnings('default', category=NaturalNameWarning)
        try:
            clean_data_fraction = os.path.getsize(filepath)/os.path.getsize(tmp_h5)
        except FileNotFoundError:
            print("No data remain - this will show as FileNotFoundError. The initial",
                  "file will be restored. Delete it, if you're sure about it.")
            raise
        if not force and clean_data_fraction < .67:
            raise Exception(f"Only {clean_data_fraction:%} of the original file size remain. If this seems plausible to you, rerun setting `force=True`.")
    except:
        print("Restoring original (potentially corrupt) file because error occurred.")
        if os.path.isfile(filepath):
            os.remove(filepath)
        shutil.move(tmp_h5, filepath)
        print("Successfully restored initial state. Reraising error:")
        raise
    else:
        print("Reperation was successful: removed duplicates and sorted index.")
    finally:
        if os.path.isfile(tmp_h5):
            os.remove(tmp_h5)
__all__.append("repair_l2_cache")


def rgi_code_translator(input: str, out_type: str = "full_name") -> str:
    if isinstance(input, list):
        return [rgi_code_translator(element, out_type) for element in input]
    if isinstance(input, int) or len(input) <= 2 and int(input) < 20:
        return rgi_o1region_translator(input, out_type)
    if re.match(r"\d\d-\d\d", input):
        return rgi_o2region_translator(*[int(x) for x in input.split("-")], out_type=out_type)
    raise ValueError(f"Input {input} not understood. Pass RGI o1- or o2region codes.")
__all__.append("rgi_code_translator")


def rgi_o1region_translator(input: int, out_type: str = "full_name") -> str:
    if isinstance(input, list):
        return [rgi_o1region_translator(element, out_type) for element in input]
    lut = pd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o1regions.feather"),
                          columns=["o1region", "full_name", "long_code"],
                          ).set_index("o1region")
    return lut.loc[f"{input:02d}", out_type]
__all__.append("rgi_o1region_translator")


def rgi_o2region_translator(o1: int, o2: int, out_type: str = "full_name") -> str:
    if isinstance(o1, list):
        return [rgi_o2region_translator(o1_, o2_, out_type) for o1_, o2_ in zip(o1, o2)]
    if isinstance(o2, list):
        return [rgi_o2region_translator(o1, o2_, out_type) for o2_ in o2]
    if o1 == 5 and o2 in range(11, 16):
        if out_type != "full_name":
            raise NotImplementedError()
        return dict([(11, "North Greenland"), (12, "West Greenland"), (13, "South West Greenland"), (14, "South East Greenland"), (15, "East Greenland")])[o2]
    lut = pd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"),
                          columns=["o2region", "full_name", "long_code"],
                          ).set_index("o2region")
    return lut.loc[f"{o1:02d}-{o2:02d}", out_type]
__all__.append("rgi_o2region_translator")


def weighted_mean_excl_outliers(df: pd.DataFrame|xr.Dataset = None,
                                weights: np.ndarray|str = "weights", *,
                                values: np.ndarray|str = None,
                                deviation_factor: int = 5,
                                return_mask: bool = False,
                                ) -> float:
    """Calculates the weighted average after excluding outliers.

    Note: This function uses `np.average` which expects weights similar
          to 1/variance - incontrast to `np.lstsq` and derivates, that
          expect 1/std and square the weights internally.

    Args:
        df (DataFrame): DataFrame containing values and weights.
        values (1d-numpy array): Values to average or name of dataframe
            column to average.
        weights (1d-numpy array): Weights to apply to values or name
            of dataframe column to use.
        deviation_factor (int, optional): Factor to apply to standard
            deviation. Values further appart from average are excluded.
            Defaults to 5.

    Returns:
        float: Weighted average excluding outliers.
        if `return_mask`: returns a boolean mask, true where outlier.
                          The mask is same as input type.
    """
    # todo: write a test: mainly confirm math works
    if isinstance(df, pd.DataFrame) or isinstance(df, xr.Dataset):
        values = df[values].values
        if isinstance(weights, str):
            weights = df[weights].values
    outlier_mask = flag_outliers(values, weights=weights, stat=np.average, deviation_factor=deviation_factor,
                                 scaling_factor=1)
    effective_sample_size = float(weights[~outlier_mask].sum()**2/(weights[~outlier_mask]**2).sum())
    # print(outlier_mask)
    if effective_sample_size > 6:
        avg = float(np.average(values[~outlier_mask], weights=weights[~outlier_mask]))
        _var = float(np.average((values[~outlier_mask]-avg)**2, weights=weights[~outlier_mask]))
        if return_mask:
            return avg, _var, effective_sample_size, outlier_mask
    else:
        avg = np.nan
        _var = np.nan
        if return_mask:
            return avg, _var, effective_sample_size, outlier_mask == -1 # all False
    return avg, _var, effective_sample_size
__all__.append("weighted_mean_excl_outliers")


def xycut(data: gpd.GeoDataFrame, x_chunk_meter = 3*4*5*1_000, y_chunk_meter = 3*4*5*1_000)\
    -> list[dict[str, Union[float, gpd.GeoDataFrame]]]:
    # 3*4*5=60 [km] fits many grid cell sizes and makes reasonable chunks
    # ! only implemented for l2 data. however, easily convertible for l1b and l3 data

    def lower_x(x):
        return (x//x_chunk_meter)*x_chunk_meter
    def lower_y(y):
        return (y//y_chunk_meter)*y_chunk_meter
    minx, miny, maxx, maxy = data.total_bounds
    chunks = []
    for x in np.arange(lower_x(minx), lower_x(maxx)+1, x_chunk_meter):
        for y in np.arange(lower_y(miny), lower_y(maxy)+1, y_chunk_meter):
            tmp = data.cx[x:x+x_chunk_meter,y:y+y_chunk_meter]
            if tmp.empty: continue
            chunks.append(dict(x_interval_start = x,
                               x_interval_stop = x + x_chunk_meter,
                               y_interval_start = y,
                               y_interval_stop = y + y_chunk_meter,
                               data = tmp))
    return chunks
__all__.append("xycut")


__all__ = sorted(__all__)
