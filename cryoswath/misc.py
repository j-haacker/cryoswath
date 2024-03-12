import ftplib
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import Geod
import re
import requests
from scipy.constants import speed_of_light
import scipy.stats
import shapely
import warnings
import xarray as xr

## Paths ##############################################################
data_path = os.path.join(os.path.dirname(__file__), "..", "data")
aux_path = os.path.join(data_path, "auxiliary")
cs_ground_tracks_path = os.path.join(aux_path, "CryoSat-2_SARIn_ground_tracks.feather")
rgi_path = os.path.join(aux_path, "RGI")
dem_path = os.path.join(aux_path, "DEM")

## Config #############################################################
WGS84_ellpsoid = Geod(ellps="WGS84")
# The following is advised to set for pandas<v3 (default for later versions)
pd.options.mode.copy_on_write = True

## Constants ##########################################################
antenna_baseline = 1.1676
Ku_band_freq = 13.575e9
sample_width = speed_of_light/(320e6*2)/2

## Functions ##########################################################

def cs_id_to_time(cs_id: str) -> pd.Timestamp:
    return pd.to_datetime(cs_id, format="%Y%m%dT%H%M%S")


def cs_time_to_id(time: pd.Timestamp) -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def download_file(url: str, out_path: str = ".") -> str:
    # snippet adapted from https://stackoverflow.com/a/16696317
    # authors: https://stackoverflow.com/users/427457/roman-podlinov
    #      and https://stackoverflow.com/users/12641442/jenia
    local_filename = os.join(out_path, url.split('/')[-1])
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename


def find_region_id(region_outlines: shapely.Polygon|shapely.MultiPolygon|gpd.GeoSeries|gpd.GeoDataFrame):
    if isinstance(region_outlines, gpd.GeoDataFrame):
        region_outlines = region_outlines.geometry
    if isinstance(region_outlines, gpd.GeoSeries):
        region_outlines = region_outlines.to_crs(4326).unary_union
    rgi_o2_gpdf = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
    return rgi_o2_gpdf[rgi_o2_gpdf.contains(region_outlines.centroid)]["long_code"].values[0]
    # ! tbi: if only small region/one glacier, make get its
    # to_planar = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3413))
    # if shapely.ops.transform(to_planar.transform, region_outlines).area > 


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
                    raise("Unkown flag: {2**i}! This points to a bug either in the code or in the data!")
        return flag_list
    else:
        flag_dictionary = pd.Series(data=cs_l1b_flag.attrs["flag_meanings"].split(" "),
                                    index=cs_l1b_flag.attrs["flag_values"])
        return flag_dictionary.loc[int(cs_l1b_flag.values)]


def gauss_filter_DataArray(da, dim, window_extent, std):
    # force window_extent to be uneven to ensure center to be where expected
    half_window_extent = window_extent//2
    gauss_weights = scipy.stats.norm.pdf(np.arange(-half_window_extent, half_window_extent+1), scale=std)
    gauss_weights = xr.DataArray(gauss_weights/np.sum(gauss_weights), dims=["window_dim"])
    if np.iscomplexobj:
        helper = da.rolling({dim: window_extent}, center=True).construct("window_dim").dot(gauss_weights)
        return helper/np.abs(helper)
    else:
        return da.rolling({dim: window_extent}, center=True).construct("window_dim").dot(gauss_weights)


def load_cs_full_file_names(update: bool = False) -> pd.Series:
    file_names_path = os.path.join(aux_path, "CryoSat-2_SARIn_file_names.pkl")
    if update:
        file_names = pd.read_pickle(file_names_path).sort_index()
        last_lta_idx = file_names[(fn[3:7]=="LTA_" for fn in file_names)].index[-1]
        print(last_lta_idx+pd.offsets.MonthBegin(-1, normalize=True))
        with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
            ftp.login(passwd="your@email.address")
            ftp.cwd("/SIR_SIN_L1")
            for year in ftp.nlst():
                if year < str(last_lta_idx.year):
                    print("skip", year)
                    continue
                try:
                    ftp.cwd(f"/SIR_SIN_L1/{year}")
                    print(f"entered /SIR_SIN_L1/{year}")
                    for month in ftp.nlst():
                        if pd.to_datetime(f"{year}-{month}") < last_lta_idx+pd.offsets.MonthBegin(-1, normalize=True):
                            print("skip", month)
                            continue
                        ftp.cwd(f"/SIR_SIN_L1/{year}/{month}")
                        print(f"scanning /SIR_SIN_L1/{year}/{month}")
                        for remote_file in ftp.nlst():
                            if remote_file[-3:] == ".nc":
                                remote_idx = pd.to_datetime(remote_file[19:34])
                                if remote_idx in file_names.index \
                                and (file_names.loc[remote_idx][3:7]=="LTA_" or remote_file[3:7]=="OFFL"):
                                    continue
                                file_names.loc[remote_idx] = remote_file[:-3]
                except (KeyboardInterrupt, Exception) as err:
                    if isinstance(err, KeyboardInterrupt):
                        file_names.to_pickle(file_names_path)
                        raise
                    warnings.warn(f"Error occurred in remote directory /SIR_SIN_L1/{year}/{month}.")
    elif os.path.isfile(file_names_path):
        return pd.read_pickle(file_names_path)
    else:
        file_names = pd.Series(name="cs_full_file_names")
        with ftplib.FTP("science-pds.cryosat.esa.int") as ftp:
            ftp.login(passwd="your@email.address")
            ftp.cwd("/SIR_SIN_L1")
            for year in ftp.nlst():
                try:
                    ftp.cwd(f"/SIR_SIN_L1/{year}")
                    print(f"entered /SIR_SIN_L1/{year}")
                    for month in ftp.nlst():
                        ftp.cwd(f"/SIR_SIN_L1/{year}/{month}")
                        print(f"scanning /SIR_SIN_L1/{year}/{month}")
                        for remote_file in ftp.nlst():
                            if remote_file[-3:] == ".nc":
                                file_names.loc[pd.to_datetime(remote_file[19:34])] = remote_file[:-3]
                except (KeyboardInterrupt, Exception) as err:
                    if isinstance(err, KeyboardInterrupt):
                        file_names.to_pickle(file_names_path)
                        raise
                    warnings.warn(f"Error occurred in remote directory /SIR_SIN_L1/{year}/{month}.")
    file_names.to_pickle(file_names_path)
    return file_names


def load_cs_ground_tracks() -> gpd.GeoDataFrame:
    cs_tracks = gpd.read_feather(cs_ground_tracks_path).set_index("index").sort_index()
    cs_tracks.index = pd.to_datetime(cs_tracks.index)
    return cs_tracks.set_crs(4326)


def load_o1region(o1code: str) -> gpd.GeoDataFrame:
    rgi_files = os.listdir(rgi_path)
    for file in rgi_files:
        if re.match(f"RGI2000-v7\.0-G-{o1code}_.*", file):
            file_path = os.path.join(rgi_path, file)
            if file[-8:] == ".feather":
                o1region = gpd.read_feather(file_path)
            elif file[-4:] == ".shp" or os.path.isdir(file_path):
                o1region = gpd.read_file(file_path)
            else:
                continue
            break
    if "o1region" not in locals():
        print("Make sure RGI files are available in data/auxiliary/RGI. If you did not download them already, you can find them at https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G/. Mind that you need to unzip them. If you decide to put them into a directory, name it as the file is named (e.g. RGI2000-v7.0-G-01_alaska).")
        raise FileNotFoundError
    return o1region


def load_o2region(o2code: str) -> gpd.GeoDataFrame:
    o1region = load_o1region(o2code[:2])
    return o1region[o1region["o2region"]==o2code]


# make contents accessible
__all__ = ["aux_path", "data_path", "dem_path", "cs_ground_tracks_path", "rgi_path", # paths ..........................
           "WGS84_ellpsoid", "antenna_baseline", "Ku_band_freq", "sample_width",     # vars ...........................
           "speed_of_light",
           "cs_id_to_time", "cs_time_to_id", "find_region_id", "flag_translator",    # funcs ..........................
           "gauss_filter_DataArray", "load_cs_full_file_names", "load_cs_ground_tracks", "load_o1region",
           "load_o2region",
           ]
