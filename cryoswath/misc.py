from configparser import ConfigParser
from contextlib import contextmanager
from dateutil.relativedelta import relativedelta
from defusedxml.ElementTree import fromstring as ET_from_str
import fnmatch
import ftplib
import geopandas as gpd
from git.repo import Repo
import glob
import h5py
import inspect
import numpy as np
import os
from packaging.version import Version
import pandas as pd
from pyproj import CRS, Geod
import queue
import rasterio
import re
from scipy.constants import speed_of_light
import scipy.stats
from scipy.stats import norm, median_abs_deviation
from scipy.stats import t as student_t
import shapely
import shutil
from sklearn import covariance, linear_model, preprocessing
import sys
from tables import NaturalNameWarning
import time
import threading
import traceback
from typing import Union
import warnings
import xarray as xr

from . import gis

# make contents accessible
__all__ = [  # variables
    "antenna_baseline",
    "cryosat_id_pattern",
    "Ku_band_freq",
    "nanoseconds_per_year",
    "sample_width",
    "speed_of_light",
    "WGS84_ellpsoid",
]

__all__.extend([  # functions
    "cs_id_to_time",
    "cs_time_to_id",
    "define_elev_band_edges",
    "discard_frontal_retreat_zone",
    "find_region_id",
    "flag_translator",
    "ftp_cs2_server",
    "gauss_filter_DataArray",
    "get_dem_reader",
    "interpolate_hypsometrically",
    "load_basins",
    "load_cs_full_file_names", 
    "load_cs_ground_tracks",
    "load_o1region",
    "load_o2region",
])

__all__.extend([  # patches
    "monkeypatch",
    "patched_xr_decode_tDel",
    "patched_xr_decode_scaling",
])


def init_project():
    if (
        os.path.exists("data")
        or os.path.exists("scripts")
    ):
        Exception("Make sure \"data\" and \"scripts\" do not exist in your working directory.")
    try:    
        Repo.clone_from("https://github.com/j-haacker/cryoswath.git", "data", branch="data")
        Repo.clone_from("https://github.com/j-haacker/cryoswath.git", "scripts", branch="scripts")
    except:
        os.makedirs("scripts")
    config_file = os.path.join("scripts", "config.ini")
    config = ConfigParser()
    if os.path.isfile(config_file):
        config.read(os.path.join("scripts", "config.ini"))
    config["path"] = {"base": os.getcwd()}
    with open(config_file, "w") as f:
        config.write(f)


## Paths ##############################################################
if os.path.isfile("config.ini"):
    config = ConfigParser()
    config.read("config.ini")
    data_path = os.path.join(config["path"]["base"], "data")
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

    __all__.extend([  # pathes
        "aux_path",
        "cs_ground_tracks_path",
        "data_path",
        "dem_path",
        "l1b_path",
        "l2_swath_path",
        "l2_poca_path",
        "l3_path",
        "l4_path",
        "rgi_path",
        "tmp_path",
    ])

else:
    warnings.warn("Failed to define path variables. You will not be able to use many cryoswath "
                  "functions. Make sure have run `cryoswath-init` and your working directory"
                  " is \"scripts\".")

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
    """Helper class to download via ftp.
    """
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
        """Appends to cache.

        Args:
            new_part (binary): New part.
        """
        self._cache.extend(new_part)
    __all__.append("add")
__all__.append("binary_chache")


def cs_id_to_time(cs_id: str) -> pd.Timestamp:
    """Formats CryoSat-2 file time tag as timestamp.

    Args:
        cs_id (str): CryoSat-2 file time tag.

    Returns:
        pd.Timestamp: Timestamp.
    """
    return pd.to_datetime(cs_id, format="%Y%m%dT%H%M%S")


def cs_time_to_id(time: pd.Timestamp) -> str:
    """Converts timestamp to CryoSat-2 file time tag.

    Args:
        time (pd.Timestamp): Timestamp.

    Returns:
        str: CryoSat-2 file time tag.
    """
    return time.strftime("%Y%m%dT%H%M%S")


def convert_all_esri_to_feather(dir_path: str = None) -> None:
    """Converts ESRI/ArcGIS formatted files to feathers

    Finds all .shp in given directory. Not recursive.

    Args:
        dir_path (str, optional): Root directory. Defaults to None.
    """
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


def define_elev_band_edges(elevations: xr.DataArray) -> np.ndarray:
    elev_range_80pctl = float(elevations.quantile([.1, .9]).diff(dim="quantile").values.item(0))
    if elev_range_80pctl >= 500:
        elev_bin_width = 50
    else:
        elev_bin_width = elev_range_80pctl/10
    return np.arange(elevations.min(), elevations.max()+elev_bin_width, elev_bin_width)


def discard_frontal_retreat_zone(
        ds,
        replace_vars: list,
        main_var: str = "_median",
        elev: str = "ref_elev",
        mode: str = None,
        threshold: float = None
) -> xr.Dataset:
    """Unsets values in zone of frontal retreat

    Areas that are not continuesly glacierized distort the fitted
    polynomial that is used to fill voids, biasing aggregates of later
    products.

    This function compares the change rates in lower elevation bands. If
    the lowest bands show smaller changes than those immediately above
    them, this is interpreted as indication of a temporarily
    glacier-free surface.

    Args:
        ds (_type_): _description_
        replace_vars (list): _description_
        main_var (str, optional): _description_. Defaults to "_median".
        elev (str, optional): _description_. Defaults to "ref_elev".
        mode (str, optional): _description_. Defaults to None.
        threshold (float, optional): _description_. Defaults to None.

    Returns:
        xr.Dataset: _description_
    """
    
    if mode is None:
        if "time" in ds:
            mode = "temporal"
        else:
            mode = "trend"
    
    if threshold is None:
        if mode == "temporal":
            threshold = 10
        elif mode == "trend":
            threshold = 1
        else:
            ValueError("Value for 'mode' not allowed.")

    def custom_count(data, **kwargs):
        return ((~np.isnan(data)).sum(0)>5).sum()>4
    
    def median_mad(data, **kwargs):
        return np.nanmedian(median_abs_deviation(data, 0, **kwargs))

    try:
        # print(define_elev_band_edges(ds[elev]))
        bands = ds[main_var].groupby_bins(ds[elev], define_elev_band_edges(ds[elev])[:5], include_lowest=True)
    except ValueError as err:
        if str(err) == "arange: cannot compute length":
            return ds
        raise

    if mode == "temporal":
        if (
            (ds[main_var].count("time") > 5).sum() < 5
             or not bands.reduce(custom_count, ...).all()
        ):
            return ds
        tmp = bands.reduce(median_mad, ..., nan_policy="omit")
    else:
        if (
            ds[main_var].count() < 5
            or not (bands.count() > 4).all()
        ):
            # print(ds[main_var].count(), bands.count())
            return ds
        tmp = np.abs(bands.mean())
    
    if not (tmp > threshold).any():
        # print(ds.basin_id.values.item(0), "too small.", ds[elev].count().values.item(0), "cells in total")
        return ds

    front_bin = ((tmp > tmp.max() / 2).cumsum() != 0).idxmax().values.item(0)

    # # debugging:
    # import matplotlib.pyplot as plt
    # tmp.plot()
    # plt.show()

    if isinstance(replace_vars, str):
        replace_vars = [replace_vars]
    for var_ in replace_vars:
        ds[var_] = xr.where(ds[elev] < front_bin.left, np.nan, ds[var_])
    
    return ds


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


def extend_filename(file_name: str, extension: str) -> str:
    """Adds string at end of file name, before last "."

    Args:
        file_name (str): File name or path.
        extension (str): String to insert at end.

    Returns:
        str: As input, including extension.
    """
    fn_parts = file_name.split(os.path.extsep)
    return os.path.extsep.join(fn_parts[:-1]) + extension + os.path.extsep + fn_parts[-1]
__all__.append("extend_filename")


# ! make recursive
def filter_kwargs(func: callable,
                  kwargs: dict, *,
                  blacklist: list[str] = None,
                  whitelist: list[str] = None,
                  ) -> dict:
    """Automatically reduces dict to accepted inputs

    Detects expected key-word arguments of a function and only passes
    those. Use black- and whitelists to refine.

    Args:
        func (callable): Target function.
        kwargs (dict): KW-args to be filtered.
        blacklist (list[str], optional): Blacklist undesired arguments.
            Defaults to None.
        whitelist (list[str], optional): Include extra arguments, that are
            not part of the functions signature. Defaults to None.

    Returns:
        dict: Filtered kw-args.
    """
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
    """Returns RGI id for multitude of inputs

    Special behavior in Greenland! If o2 region is requested, return id of
    "custom" subregion: 05-11--05-15 for N, W, SW, SE, E. See geo-feathers
    in `data/auxiliary/RGI/05-1*.feather`.

    Args:
        location (any): Can be a geo-referenced xarray.DataArray, a
            geopandas.GeoDataFrame or Series, or a shapely.Geometry.
        scope (str, optional): One of "o1", "o2", or "basin". Defaults to
            "o2".

    Raises:
        Exception: `scope` is "o2" and `location` is in Greenland but
            - not in one of the custom subregions or
            - in more than one custom subregion.

    Returns:
        str: RGI id.
    """
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
    """Flags data that is considered outlier given a set of assumptions

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


@contextmanager
def ftp_cs2_server(**kwargs):
    try:
        config = ConfigParser()
        config.read("config.ini")
        email = config["user"]["email"]
    except KeyError:
        print("\n\nPlease call `misc.update_email()` to provide your email address as ESA asks",
              "for it as password when downloading data via ftp.\n\n")
        raise
    with ftplib.FTP("science-pds.cryosat.esa.int", **kwargs) as ftp:
        ftp.login(passwd=email)
        yield ftp
    

def gauss_filter_DataArray(da: xr.DataArray, dim: str, window_extent: int, std: int) -> xr.DataArray:
    """Low-pass filters input array.

    Convolves each vector of an array along the specified dimension with a
    normalized gauss-function having the specified standard deviation.

    Args:
        da (xr.DataArray): Data to be filtered.
        dim (str): Dimension to apply filter along.
        window_extent (int): Window width. If not uneven, it is increased.
        std (int): Standard deviation of gauss-filter.

    Returns:
        xr.DataArray: _description_
    """
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
    """Determines which DEM to use

    Attempts to determine location of `data` and returns appropriate
    `rasterio.io.DatasetReader`. Only implemented for ArcticDEM and
    REMA.

    Args:
        data (any): Defaults to None.

    Raises:
        NotImplementedError: If region can't be inferred.

    Returns:
        rasterio.DatasetReader: Reader pointing to the file.
    """

    raster_extensions = ['tif', 'nc']

    if isinstance(data, shapely.Geometry):
        lat = np.mean(data.bounds[1::2])
    elif isinstance(data, float) \
            or isinstance(data, int) \
            or (isinstance(data, np.ndarray) and data.size==1):
        lat = data
    elif "lat_20_ku" in data:
        lat = data.lat_20_ku.values[0]
    elif isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
        lat = np.mean(data.rio.transform_bounds("EPSG:4326")[1::2])
    elif isinstance(data, gpd.GeoSeries) or isinstance(data, gpd.GeoDataFrame):
        lat = data.to_crs(4326).union_all("coverage").centroid.y
    elif isinstance(data, str):
        if data.lower() in ["arctic", "arcticdem"]:
            lat = 90
        elif data.lower() in ["antarctic", "rema"]:
            lat = -90
        elif os.path.sep in data:
            return rasterio.open(data)
        elif any([data.split(".")[-1] in raster_extensions]):
            return rasterio.open(os.path.join(dem_path, data))
    if "lat" not in locals():
        raise NotImplementedError(f"`get_dem_reader` could not handle the input of type {data.__class__}. See doc for further info.")
    if lat > 0:
        # return rasterio.open(os.path.join(dem_path, "arcticdem_mosaic_100m_v4.1_dem.tif"))
        dem_filename = "arcticdem_mosaic_100m_v4.1_dem.tif"
    else:
        dem_filename = "rema_mosaic_100m_v2.0_filled_cop30_dem.tif"
    if not os.path.isfile(os.path.join(dem_path, dem_filename)):
        raster_file_list = []
        for ext in raster_extensions:
            raster_file_list.extend(glob.glob('*.'+ext, root_dir=dem_path))
        print("DEM not found with default filename. Please select from the following:\n",
              ", ".join(raster_file_list), flush=True)
        dem_filename = input("Enter filename:")
    return rasterio.open(os.path.join(dem_path, dem_filename))



def interpolate_hypsometrically(ds: xr.Dataset,
                                main_var: str,
                                error: str,
                                elev: str = "ref_elev",
                                weights: str = "weights",
                                outlier_replace: bool = False,
                                outlier_limit: float = 2,
                                return_coeffs: bool = False,
                                fit_sanity_check: dict = None,
                                ) -> xr.Dataset:
    """Fills data gaps by hypsometrical interpolation

    If sufficient data is provided, this routine sorts and bins the data
    by elevation bands and fits a third-order polynomial to the weighted
    averages.

    Sufficient data requires 4 or more bands, with an effective sample
    size of 6 or larger, that span at least 2/3 of the total elevation
    range. The weights used to calculate the weighted average are the
    reciprocal squared errors if no weights are provided.

    If dimension "time" exists, recurse into time steps and interpolate
    per time step.

    Args:
        ds (xr.Dataset): Input with voids. The input has to be along
            dimension "stacked_x_y".
        main_var (str): Name of variable to interpolate. error (str):
            Name of errors. Where interpolated, errors will be filled by
            the scaled RMSE of the fit. The scaling factor will be
            inferred from `error`! Include one of "std", "iqr", "mad",
            "95" in the `error` variable name. If non can be found, it
            is assumed to be the standard deviation ("std"). The error
            data are only used if weights are not provided.
        elev (str, optional): Name of variable that contains the
            reference elevation used for binning. If the variable does
            not exist, it is attempted to read the reference elevations
            from disk. Defaults to "ref_elev".
        weights (str, optional): Provide name of variable that contains
            the weights. The weights will be passed to `numpy.average`
            and should be 1/variance or similar. Defaults to "weights".
        outlier_replace (bool, optional): If enabled, also interpolates
            outliers. Defaults to False.
        outlier_limit (float, optional): Factor of outlier scale (e.g.
            standard deviation). Defaults to 2.
        return_coeffs (bool, optional): If enabled, also returns 3rd
            order polynomial parameters in `numpy.polyfit
            <https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html>`_
            order (highest to lowest). Defaults to False.
        fit_sanity_check (dict, optional): Defaults to None. If None or
            False, it will not be used. If you want to test the
            polynomial gradients, either set to True or pass a `dict`.
            If True, default values will be used; that are 0.1 if used
            with elevation difference to a ref. DEM and 0.05 if used
            with elevation change trends. If you pass a `dict`, the key
            "max_allowed_gradient" will be used. If the gradient is
            steeper than the threshold the model is rejected.
    Returns:
        xr.Dataset: Filled dataset.
    """

    def select_returns(return_coeffs, ds, coeffs):
        if return_coeffs: return ds, coeffs
        else: return ds

    def design_matrix(x_vals):
        return np.hstack([x_vals, x_vals**2, x_vals**3])

    def invert_3rd_order_coeff_scaling(scaler, coeffs):
        mu, sig = scaler.mean_[0], scaler.scale_[0]
        p0, p1, p2, p3 = coeffs/np.hstack([1, design_matrix(sig)])[::-1]
        return np.array([
            p0,
            p1-3*mu*p0,
            p2-2*mu*p1+3*mu**2*p0,
            p3-mu*p2+mu**2*p1-mu**3*p0
        ])

    if "time" in ds.dims and len(ds.time) > 1:
        # note: `groupby("time")` creates time depencies for all data_vars. this
        #       requires taking note of those data_vars that do not depend on
        #       time and reset those after the operation
        no_time_dep = [data_var for data_var in ds.data_vars if not "time" in ds[data_var].dims]
        if fit_sanity_check == True:
            # set default sanity check for elevation differences wrt. ref. DEM
            fit_sanity_check = {"max_allowed_gradient": 10/100}  # [10 m elev.diff. per 100 m of elevation]
        ds = ds.groupby("time", squeeze=False).map(interpolate_hypsometrically, main_var=main_var, elev=elev, error=error, outlier_replace=outlier_replace, return_coeffs=return_coeffs, fit_sanity_check=fit_sanity_check)
        for var_name in no_time_dep:
            ds[var_name] = ds[var_name].isel(time=0)
        return select_returns(return_coeffs, ds, np.array([np.nan]*4))
    else:
        if fit_sanity_check == True:
            # set default sanity check for elevation change rate
            fit_sanity_check = {"max_allowed_gradient": 5/100}  # [10 m/yr elev.trend per 100 m of elevation]
    
    # this function uses boolean indexing which is not possible with dask
    # arrays. so if ds contains dask arrays, compute them.
    if ds.chunks is not None:
        ds = ds.compute()

    # tbi: currently the data needs to have the single dimension "stacked_x_y".
    #      implement automatic stacking and remove the hard coded dim name.

    # below might need fill_missing_coords. naively: should not be important
    neighbours = ds[main_var].unstack().sortby("x").sortby("y").rolling(x=5, y=5, min_periods=3, center=True)
    neighbour_count = neighbours.count().stack(stacked_x_y=["x", "y"]).reindex_like(ds[main_var])
    neighbour_mean = neighbours.mean().stack(stacked_x_y=["x", "y"]).reindex_like(ds[main_var])
    if outlier_replace:
        neighbour_elev = ds[elev].unstack().sortby("x").sortby("y").rolling(x=5, y=5, min_periods=3, center=True)
        neighbour_elev_mean = neighbour_elev.mean().stack(stacked_x_y=["x", "y"]).reindex_like(ds[main_var])
        neighbour_std = neighbours.std().stack(stacked_x_y=["x", "y"]).reindex_like(ds[main_var])
        neighbour_elev_std = neighbour_elev.std().stack(stacked_x_y=["x", "y"]).reindex_like(ds[main_var])
        noise = (np.abs(ds[main_var]-neighbour_mean)/neighbour_std - np.abs(ds[elev]-neighbour_elev_mean)/neighbour_elev_std) > outlier_limit
        # print(neighbour_count>=6, noise)
        ds[main_var] = xr.where(np.logical_and(neighbour_count>=6, noise), np.nan, ds[main_var])
        # # debugging plot:
        # import matplotlib.pyplot as plt
        # noise.astype("int").unstack().sortby("x").sortby("y").T.plot(cmap="cool")
        # plt.show()
    # # if the reference elevations contain nan values, this leads to errors
    # index_with_nan_in_elev = ds[ds[elev].dims[0]]
    # ds = ds.where(~ds[elev].isnull()).dropna(ds[elev].dims[0])
    # assign weights if not present. use previously assigned weights to
    # prevent using previously filled cells from inform a new average.
    if weights not in ds:
        ds[weights] = 1/ds[error]**2
    # abort if too little data (checking elevation and data validity).
    # necessary to prevent errors but also introduces data gaps
    if ds[elev].where(ds[error]>0).count() < 24:
        # print("too little data")
        return select_returns(return_coeffs, ds, np.array([np.nan]*4))
    # also, abort if there isn't anything to do
    if not ds[error].isnull().any() and not outlier_replace:
        # print("nothing to do")
        return select_returns(return_coeffs, ds, np.array([np.nan]*4))
    group_obj = ds.groupby_bins(elev, define_elev_band_edges(ds[elev]), include_lowest=True)
    elev_bin_means = pd.Series(index=group_obj.groups)
    elev_bin_errs = pd.Series(index=group_obj.groups)
    fill_mask = -1*xr.ones_like(ds[main_var])
    for label, group in group_obj:
        if (group[weights] > 0).sum() < 6:
            continue
        vals = group[main_var].fillna(-9999).squeeze() # make it obvious if anything goes wrong
        w = group[weights].fillna(0).squeeze()
        avg, _var, effective_samp_size, to_be_filled_mask = weighted_mean_excl_outliers(
            values=vals, weights=w, deviation_factor=outlier_limit, return_mask=True)
        err = (_var/effective_samp_size)**.5 
        if outlier_replace:
            to_be_filled_mask = np.logical_or(group[main_var].isnull().squeeze(), to_be_filled_mask)
        else:
            to_be_filled_mask = group[main_var].isnull().squeeze()
        to_be_filled_mask = xr.align(fill_mask, to_be_filled_mask, join="left", fill_value=-1)[1]
        fill_mask = xr.where(to_be_filled_mask!=-1, to_be_filled_mask, fill_mask)
        if np.isnan(avg):
            # print("calc weighted avg failed (probably insufficient data)", label)
            continue
        # # debugging notice
        # print("calc weighted avg succeeded", label)
        # print("avg", avg, "_var", _var, "effective_samp_size", effective_samp_size, "err", err)
        elev_bin_means.loc[label] = avg
        elev_bin_errs.loc[label] = err
    elev_bin_means.dropna(inplace=True)
    elev_bin_errs.dropna(inplace=True)
    # print(elev_range_80pctl)
    if (
        elev_bin_means.empty
        or len(elev_bin_means.index) < 5
    ):
        print("data doesn't cover sufficient elevation bands", elev_bin_means)
        return select_returns(return_coeffs, ds, np.array([np.nan]*4))
    ## fit polynomial
    # print(elev_bin_means, elev_bin_errs)
    try:
        x_vals = np.array([[idx.mid for idx in elev_bin_means.index]]).T
        scaler = preprocessing.StandardScaler().fit(x_vals, sample_weight=1/elev_bin_errs.values)
        # print(x_vals, scaler.transform(x_vals))
        # print(x_vals, design_matrix(x_vals), elev_bin_means.values, 1/elev_bin_errs.values)
        # cov = covariance.EmpiricalCovariance().fit(design_matrix(scaler.transform(x_vals))).covariance_
        fit = linear_model.Ridge(1, solver="svd").fit(design_matrix(scaler.transform(x_vals)), elev_bin_means.values, 1/elev_bin_errs.values)
        coeffs = np.hstack((fit.intercept_, fit.coef_))[::-1]
    except np.linalg.LinAlgError: # not sure what error sklearn raises
        print(elev_bin_means)
        print(elev_bin_errs)
        return select_returns(return_coeffs, ds, np.array([np.nan]*4))
    if fit_sanity_check is not None:
        if (
            np.abs(np.polyval(np.polyder(coeffs),
                              scaler.transform(np.array([elev_bin_means.index[0].mid,
                                                         elev_bin_means.index[-1].mid])[:,None]))).max()
            > fit_sanity_check["max_allowed_gradient"]*scaler.var_**.5
        ):
            # warnings.warn("discarding fit because unrealistic - !note: this is usually not the"
            #               + "desired behavior const./linear extrapolation is used instead of"
            #               + "the fit which renders this check obsolete!")
            return select_returns(return_coeffs, ds, np.array([np.nan]*4))
    def scale(x):
        if isinstance(x, xr.DataArray):
            x = x.values
        elif isinstance(x, (int, float, list)):
            x = np.array(x)
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        return scaler.transform(x)
    def const_extrapol(data, pivot):
        return np.polyval(coeffs, scale(pivot).flatten()) + xr.zeros_like(data)
    def linear_extrapol(data, pivot):
        return (np.polyval(np.polyder(coeffs), scale(pivot))*(scale(data)-scale(pivot))).flatten() + const_extrapol(data, pivot)
    extrap_below = ds[elev] < elev_bin_means.index[0].mid
    extrap_above = ds[elev] > elev_bin_means.index[-1].mid
    modelled_list = [xr.DataArray(fit.predict(design_matrix(scale(ds[elev]))), coords={"stacked_x_y": ds.stacked_x_y}, dims="stacked_x_y")[~np.logical_or(extrap_below, extrap_above)]]
    if extrap_below.any():
        modelled_list.append(linear_extrapol(ds[elev][extrap_below], elev_bin_means.index[0].mid))
    if extrap_above.any():
        modelled_list.append(linear_extrapol(ds[elev][extrap_above], elev_bin_means.index[-1].mid))
    modelled = xr.concat(modelled_list, "stacked_x_y").reindex_like(ds[main_var]).astype(ds[main_var].dtype)
    fit_x_range = elev_bin_means.index[-1].mid - elev_bin_means.index[0].mid
    modelled = xr.where(ds[elev]<elev_bin_means.index[0].mid - fit_x_range/3,
                        xr.zeros_like(ds[elev])+linear_extrapol(xr.zeros_like(ds[elev][:1])+elev_bin_means.index[0].mid - fit_x_range/3, elev_bin_means.index[0].mid).values[0],
                        modelled)
    modelled = xr.where(ds[elev]>elev_bin_means.index[-1].mid + fit_x_range/3,
                        xr.zeros_like(ds[elev])+linear_extrapol(xr.zeros_like(ds[elev][:1])+elev_bin_means.index[-1].mid + fit_x_range/3, elev_bin_means.index[-1].mid).values[0],
                        modelled)
    elev_bin_min = elev_bin_means.min()-2*elev_bin_errs.iloc[elev_bin_means.argmin()]
    elev_bin_max = elev_bin_means.max()+2*elev_bin_errs.iloc[elev_bin_means.argmax()]
    modelled = xr.where(modelled>elev_bin_max,
                        elev_bin_max,
                        modelled)
    modelled = xr.where(modelled<elev_bin_min,
                        elev_bin_min,
                        modelled)
    # # debugging plot:
    # import matplotlib.pyplot as plt
    # _, ax = plt.subplots(ncols=2, figsize=(18,6))
    # ds[main_var].unstack().sortby("x").sortby("y").T.plot(ax=ax[0], robust=True, cmap="RdYlBu")
    # modelled.unstack().sortby("x").sortby("y").T.plot(ax=ax[1], robust=True, cmap="RdYlBu")
    # plt.show()
    residuals = ds[main_var]-modelled
    # # debugging plot:
    # import matplotlib.pyplot as plt
    # (np.abs(neighbour_mean-modelled) - outlier_limit*residuals.std()/(neighbour_count/4)**.5).unstack().sortby("x").sortby("y").T.plot(robust=True, cmap="RdYlBu")
    # plt.show()
    local_deviation = np.logical_and(
        neighbour_count >= 6,
        np.abs(neighbour_mean-modelled) > outlier_limit*residuals.std()/(neighbour_count/4)**.5
    )
    modelled = xr.where(local_deviation, neighbour_mean, modelled)
    if outlier_replace:
        fill_mask = xr.where(local_deviation, 0, fill_mask)
        # std is used as a deviation measure because the fit relies on
        # normal distributed errors anyway. however, maybe it would be
        # better to use the MAD (and maybe go for some max.likelihood
        # optimizer)
        fill_mask = np.logical_or(ds[main_var].isnull(),
                                  np.logical_and(fill_mask!=0,
                                                 np.abs(residuals) > outlier_limit * residuals.std()))
    else:
        fill_mask = ds[main_var].isnull()
    # # debugging plot:
    # import matplotlib.pyplot as plt
    # plt.scatter(ds[elev].where(~fill_mask).values.flatten(), ds[main_var].values.flatten())
    # plt.scatter(ds[elev].where(fill_mask).values.flatten(), ds[main_var].values.flatten(), ec="tab:purple", fc="none")
    # # plt.scatter(ds[elev].where(ds[main_var].isnull()).values.flatten(), np.zeros(ds[elev].size), ec="tab:purple", fc="none")
    # plt.scatter(ds[elev].where(ds[main_var].isnull()).values.flatten(), modelled, ec="tab:purple", fc="none")
    # tmp_x_vals = np.linspace(ds[elev].min(), ds[elev].max(), 50)[:,None]
    # plt.plot(tmp_x_vals, fit.predict(design_matrix(scaler.transform(tmp_x_vals))), c="tab:orange")
    # plt.errorbar(x_vals, elev_bin_means, elev_bin_errs, ls="none", c="tab:red")
    # if "time" in ds:
    #     plt.title(ds.time.values)#.strftime("%Y-%m-%d")
    # plt.ylim([ds[main_var].min(), ds[main_var].max()])
    # plt.show()
    try:
        # print(fill_mask)
        ds[main_var] = xr.where(fill_mask, modelled, ds[main_var])
    except:
        # print(modelled)
        raise
    RMSE = (residuals.where(~fill_mask)**2).mean()**.5
    if "std" in error.lower():
        pass
    elif "iqr" in error.lower():
        RMSE *= 2*norm.isf(.25)
    elif "mad" in error.lower():
        RMSE *= norm.isf(.25)
    elif "95" in error.lower():
        RMSE *= norm.isf(.025)
    # print(fill_mask)
    ds[error] = xr.where(fill_mask, RMSE, ds[error])
    ds[weights] = xr.where(fill_mask, 0, ds[weights])
    # # restore data gaps
    # ds = ds.reindex_like(index_with_nan_in_elev)
    # tbi: if initially stacked, unstack here
    return select_returns(return_coeffs, ds, invert_3rd_order_coeff_scaling(scaler, coeffs))
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
    elif update == "quick":
        last_lta_idx = file_names.index[-1]
        print(last_lta_idx+pd.offsets.MonthBegin(-1, normalize=True))
    elif update == "version":
        # implement, also, to actually update the files or remove outdated - or
        # think of something to prevent that old data receives a new name
        raise Exception("Functionality to update L1b file version (e.g. ...E001.nc"
                        + "vs ...E003.nc) is not yet implemented.")
    if update in ["regular", "version"]:
        # ! "regular" should also be baseline and version aware
        last_lta_idx = file_names[(fn[3:7]=="LTA_" for fn in file_names)].index[-1]
        print(last_lta_idx+pd.offsets.MonthBegin(-1, normalize=True))

    with ftp_cs2_server() as ftp:
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
    advance_end = isinstance(end_datetime, str) and re.match(r"^20[0-9]{2}.?[01][0-9]$", end_datetime)
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    if advance_end:
        end_datetime = end_datetime + pd.DateOffset(months=1)
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
            with ftp_cs2_server() as ftp:
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
            with ftp_cs2_server() as ftp:
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
            region_of_interest = load_glacier_outlines(region_of_interest, "glaciers", union=False).geometry.values
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
    """Loads RGI v7 basin or complex outlines and meta data

    Args:
        o1code (str): starting with "01".."20"
        product (str, optional): Either "glaciers" or "complexes". Defaults to "complexes".

    Raises:
        ValueError: If o1code can't be recognized.
        FileNotFoundError: If RGI data is missing.

    Returns:
        gpd.GeoDataFrame: Queried RGI data with geometry column containing
        the outlines.
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
    """Loads RGI v7 basin or complex outlines and meta data

    Args:
        o2code (str): RGI o2 code.
        product (str, optional): Either "glaciers" or "complexes". Defaults
            to "complexes".

    Returns:
        gpd.GeoDataFrame: Queried RGI data with geometry column containing
        the outlines.
    """
    o1region = load_o1region(o2code[:2], product)
    # special handling for greenland periphery
    if o2code.startswith("05") and not o2code.endswith("01"):
        return o1region[o1region.intersects(gpd.read_feather(os.path.join(rgi_path, o2code+".feather")).union_all("coverage"))]
    return o1region[o1region["o2region"]==o2code[:5]]


def load_basins(rgi_ids: list[str]) -> gpd.GeoDataFrame:
    """Loads RGI v7 basin ~or complex~ outlines and meta data

    Args:
        rgi_ids (list[str]): RGI basin ids, all within the same RGI o1 region.

    Returns:
        gpd.GeoDataFrame: Queried RGI data with geometry column containing
        the outlines.
    """
    if len(rgi_ids) > 1:
        assert(all([id[:17]==rgi_ids[0][:17]] for id in rgi_ids))
    product_code, o1_code = rgi_ids[0].split("-")[2:4]
    rgi_o1_gpdf = load_o1region(o1_code, product="glaciers" if product_code=="G" else "complexes")
    id_to_index_series = pd.Series(data=rgi_o1_gpdf.index, index=rgi_o1_gpdf.rgi_id)
    return rgi_o1_gpdf.loc[id_to_index_series.loc[rgi_ids].values]


def load_glacier_outlines(identifier: str|list[str],
                          product: str = "complexes",
                          union: bool = True,
                          crs: int|CRS = None,
                          ) -> shapely.MultiPolygon:
    """Loads RGI v7 basin or complex outlines and meta data

    Args:
        identifier (str | list[str]): RGI id: either o1, o2, or
            basin/complex id.
        product (str, optional): Either "glaciers" or "complexes". Defaults
            to "complexes".
        union (bool, optional): For backward compatibility, if enabled (by
            default) only return union of all shapes. If disabled, return
            full GeoDataFrame. Defaults to True.
        crs (int | CRS, optional): Convenience option to reproject shape(s)
            to crs. Defaults to None.

    Raises:
        ValueError: If identifier was not understood.

    Returns:
        shapely.MultiPolygon: Union of basin shapes. If `union` is disabled,
        instead return geopandas.GeoDataFrame including the full data.
    """
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


def patch_gatekeeper(module_version: str, rules: list[dict]):
    """Checks whether a patch should be applied

    Use with a list of dict like

    [{  "version":      "2.3",
        "comperator":   operator.lt,
        "action":       "skip"},
     {  "version":      "3",
        "comperator":   operator.ge,
        "action":       "warn" }]

    Args:
        module_version (str): current version of the patched module
        rules (dict): Requires keys "comparator", "version", and
            "action".

    Returns:
        str: rules["action"] if condition is met, else None
    """
    for rule in rules:
        if rule["comperator"](Version(module_version), Version(rule["version"])):
            return rule["action"]
            

@contextmanager
def monkeypatch(dictlist: list[dict]):
    """contructs a patched context

    Patching the backend of foreign funktions quickly leads to
    inconsitencies. Using the patch only within a chosen context limits
    side effects.

    Optionally, have :func:`patch_gatekeeper` manage for which version
    to apply the patch, to warn about compatibility issues, or to raise
    an error.

    Use like:

    .. code-block:: python

        patchdicts = [{
            "module":       mod1,
            "target":       "obj1",
            "replacement":  patch1,
            "version":      base_mod1.__version__,  # optional
            "rules":        rules1  # optional
        },
        {   "module":       mod2,
            "target":       "obj2",
            "replacement":  patch2
        }]
        
        with monkeypatch(patchdicts):
            <your code>

    Args:
        dictlist (list[dict]): Requires keys "module", "target", and
            "replacement".
    """
    for d in dictlist:
        if "rules" in d:
            verdict = patch_gatekeeper(d["version"], d["rules"])
            if verdict == "skip":
                continue
            elif verdict == "raise":
                raise
            elif verdict == "warn":
                warnings.warn(f"Patch not meant for {d['module']} version {d['version']}.")
        d.update({"original": getattr(d["module"], d["target"])})
        setattr(d["module"], d["target"], d["replacement"])
    try:
        yield
    finally:
        for d in dictlist:
            if "original" in d:
                setattr(d["module"], d["target"], d["original"])


def patched_xr_decode_scaling(data, scale_factor, add_offset, dtype: np.typing.DTypeLike):
    data = data.astype(dtype=dtype, copy=True)
    if scale_factor is not None:
        data = data * scale_factor
    if add_offset is not None:
        data += add_offset
    return data


def patched_xr_decode_tDel(
        num_timedeltas, units: str, time_unit="ns"
    ) -> np.ndarray:
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64 ["s", "ms", "us", "ns"] array.
    """
    from xarray.coding.times import _netcdf_to_numpy_timeunit, _check_timedelta_range, _numbers_to_timedelta, ravel, reshape
    num_timedeltas = np.asarray(num_timedeltas)
    unit = _netcdf_to_numpy_timeunit(units)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        _check_timedelta_range(np.nanmin(num_timedeltas), unit, time_unit)
        _check_timedelta_range(np.nanmax(num_timedeltas), unit, time_unit)

    timedeltas = _numbers_to_timedelta(num_timedeltas, unit, "ns", "timedelta")
    pd_timedeltas = pd.to_timedelta(ravel(timedeltas), unit="ns")

    if np.isnat(timedeltas).all():
        empirical_unit = time_unit
    else:
        empirical_unit = pd_timedeltas.unit

    if np.timedelta64(1, time_unit) > np.timedelta64(1, empirical_unit):
        time_unit = empirical_unit

    if time_unit not in {"s", "ms", "us", "ns"}:
        raise ValueError(
            f"time_unit must be one of 's', 'ms', 'us', or 'ns'. Got: {time_unit}"
        )

    result = pd_timedeltas.as_unit(time_unit).to_numpy()
    return reshape(result, num_timedeltas.shape)


def nan_unique(data: np.typing.ArrayLike) -> list:
    """Returns unique values that are not nan.

    Args:
        data (np.typing.ArrayLike): Input data.

    Returns:
        list: List of unique values.
    """
    return [element for element in np.unique(data) if not np.isnan(element)]
__all__.append("nan_unique")


def request_workers(task_func: callable, n_workers: int, result_queue: queue.Queue = None) -> queue.Queue:
    """Creates workers and provides queue to assign work

    Args:
        task_func (callable): Task.
        n_workers (int): Number of requested workers.
        result_queue (queue.Queue, optional): Queue in which to drop
            results. Defaults to None.

    Returns:
        queue.Queue: Task queue.
    """
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


def rgi_code_translator(input: str|list[str], out_type: str = "full_name") -> str:
    """Translate o1 or o2 codes to region names

    Args:
        input (str): RGI o1 or o2 codes.
        out_type (str, optional): Either "full_name" or "long_code".
            Defaults to "full_name".

    Raises:
        ValueError: If input is not understood.

    Returns:
        str: Either full name or RGI "long_code".
    """
    if isinstance(input, list):
        return [rgi_code_translator(element, out_type) for element in input]
    if isinstance(input, int) or len(input) <= 2 and int(input) < 20:
        return rgi_o1region_translator(input, out_type)
    if re.match(r"\d\d-\d\d", input):
        return rgi_o2region_translator(*[int(x) for x in input.split("-")], out_type=out_type)
    raise ValueError(f"Input {input} not understood. Pass RGI o1- or o2region codes.")
__all__.append("rgi_code_translator")


def rgi_o1region_translator(input: int, out_type: str = "full_name") -> str:
    """Finds region name for given RGI o1 number.

    Args:
        input (int): RGI o1 number.
        out_type (str, optional): Either "full_name" or "long_code".
            Defaults to "full_name".

    Returns:
        str: Either full name or RGI "long_code".
    """
    if isinstance(input, list):
        return [rgi_o1region_translator(element, out_type) for element in input]
    lut = pd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o1regions.feather"),
                          columns=["o1region", "full_name", "long_code"],
                          ).set_index("o1region")
    return lut.loc[f"{input:02d}", out_type]
__all__.append("rgi_o1region_translator")


def rgi_o2region_translator(o1: int, o2: int, out_type: str = "full_name") -> str:
    """Finds subregion name for given RGI o1 and o2 number.

    Args:
        o1 (int): RGI o1 number.
        o2 (int): RGI o2 number.
        out_type (str, optional): Either "full_name" or "long_code".
            Defaults to "full_name".

    Returns:
        str: Either full name or RGI "long_code".
    """
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


def update_email(email):
    config = ConfigParser()
    config.read("config.ini")
    if "user" not in config:
        config["user"] = dict()
    config["user"].update({"email": email})
    with open("config.ini", "w") as f:
        config.write(f)


# CREDIT: mgab https://stackoverflow.com/a/22376126
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
__all__.append("warn_with_traceback")


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
        float: Weighted average excluding outliers. if `return_mask`,
        returns a boolean mask that is true where outliers were detected.
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
    """Chunk point data in planar reference system

    This mainly is a helper function for `l3.build_dataset()` that takes
    many data points and chunks them based on their location.
    However, it may be helpful in other contexts.

    Returns:
        list: List of dicts of which each contains the x and y extents of
        the current chunk and the GeoDataFrame or Series of the point data.
    """
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
