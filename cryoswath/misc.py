import configparser
from dateutil.relativedelta import relativedelta
from defusedxml.ElementTree import fromstring as ET_from_str
import fnmatch
import ftplib
import geopandas as gpd
import glob
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
import time
import threading
import warnings
import xarray as xr

from . import gis

# make contents accessible
__all__ = ["WGS84_ellpsoid", "antenna_baseline", "Ku_band_freq", "sample_width",     # vars ...........................
           "speed_of_light", "cryosat_id_pattern",
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
aux_path = os.path.join(data_path, "auxiliary")
cs_ground_tracks_path = os.path.join(aux_path, "CryoSat-2_SARIn_ground_tracks.feather")
rgi_path = os.path.join(aux_path, "RGI")
dem_path = os.path.join(aux_path, "DEM")
__all__.extend(["data_path",
                "l1b_path", "l2_swath_path", "l2_poca_path", "l3_path",
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

## Functions ##########################################################

def append_filename(file_name: str, appendix: str) -> str:
    fn_parts = file_name.split(os.path.extsep)
    return os.path.extsep.join(fn_parts[:-1]) + appendix + os.path.extsep + fn_parts[-1]
__all__.append("append_filename")


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
    if isinstance(location, gpd.GeoDataFrame):
        location = location.geometry
    if isinstance(location, gpd.GeoSeries):
        location = location.to_crs(4326).unary_union
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
        return rgi_region["o2region"].values[0]
    elif scope == "basin":
        rgi_glacier_gpdf = load_o2region(rgi_region["o2region"].values[0], "glaciers")
        return rgi_glacier_gpdf[rgi_glacier_gpdf.contains(location.centroid)]["rgi_id"].values[0]
    else:
        raise Exception("`scope` can be one of \"o1\", \"o2\", or \"basin\".")

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
    # # data can be either l1b.l1b_data (xarray.Dataset containing
    # # "lat_20_ku"), l2 data (geopandas.GeoDataFrame containing "lat"),
    # # or  a (lon, lat) tuple/list/dict.
    # if isinstance(data, float):
    #     lat = data
    # elif len(data.lat_20_ku) > 1:
    #     lat = np.abs(data.lat_20_ku).min()
    # else:
    #     lat = data.lat_20_ku
    # if lat
    # return rasterio.open(os.path.join(dem_path, "09-02_novaya_zemlya.tif"))
    return rasterio.open(os.path.join(dem_path, "arcticdem_mosaic_100m_v4.1_dem.tif"))


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
    # update one of: "no", "quick", "regular", "full"
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
                          nthreads: int = 8,
                          ) -> gpd.GeoDataFrame:
    """Read the GeoDataFrame of CryoSat-2 tracks from disk.

    If desired, you can query certain extents or periods by specifying
    arguments.

    Further, you can update the database by setting `update` to "regular" or
    "full". Mind that this typically takes some time (regular on the order
    of minutes, full rather hours).

    Args:
        region_of_interest (str | shapely.Polygon, optional): Can be any RGI
            code or a polygon in lat/lon (CRS EPSG:4326). Defaults to None.  
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
        nthreads (int, optional): Number of parallel ftp connections. If you
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
        last_idx = pd.Timestamp("2010-06-01")
    elif update == "regular":
        last_idx = pd.to_datetime(cs_tracks.index[-1])
    elif update != "no":
        raise ValueError("Allowed values for `update` are \"full\". \"regular\". or \"no\". "
                         +f"You set it to \"{update}\".")
    if update != "no":

        # the next two function have only a local purpose.
        def save_current_track_list(new_track_series: gpd.GeoSeries):
            """saves the tracklist; backing up the old if older than 5 days."""
            if not os.path.isfile(append_filename(cs_ground_tracks_path, "__backup"))\
                    or time.time() - os.path.getmtime(cs_ground_tracks_path)  > 5 * 24*60*60:
                print("backing up \"old\" track file")
                shutil.copyfile(cs_ground_tracks_path, append_filename(cs_ground_tracks_path, "__backup"))
            print("saving current track list to file")
            new_track_series.to_feather(cs_ground_tracks_path)

        def collect_missing_tracks(remote_files: list[str],
                                   present_tracks: gpd.GeoSeries,
                                   result_queue: queue.Queue) -> gpd.GeoSeries:
            """Gets track if not in list already.

            Args:
                files (list[str]): HDR file names. All of the same month.
                present_tracks (gpd.GeoSeries): Known tracks.
                result_queue (queue.Queue): Place to collect results from.

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
            result_queue.put(tracks_to_be_added)

        # set up and start worker-threads
        task_queue = queue.SimpleQueue()
        def worker():
            while True:
                try:
                    next_batch = task_queue.get()
                except TypeError:
                    pass
                if next_batch is not None:
                    task_queue.put(collect_missing_tracks(*next_batch))
        for i in range(nthreads):
            worker_thread = threading.Thread(target=worker, daemon=True)
            worker_thread.start()

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
            batch_size = len(remote_files)//(nthreads*3)+1
            result_queue = queue.SimpleQueue()
            while remote_files:
                try:
                    task_queue.put((remote_files[:batch_size], cs_tracks, result_queue))
                    remote_files[:batch_size] = []
                except IndexError:
                    task_queue.put((remote_files[:], cs_tracks, result_queue))
                    remote_files[:] = []
            # wait for and collect new tracks
            new_tracks_collection = []
            while not task_queue.empty() or not result_queue.empty():
                try:
                    tmp = result_queue.get(block=True, timeout=10*60)
                    new_tracks_collection.append(tmp)
                except queue.Empty:
                    print("waiting for task queue")
                    time.sleep(10)
            # append to local collection and save the result
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
    cs_tracks = cs_tracks.loc[start_datetime:end_datetime+pd.offsets.Day(1)]
    if region_of_interest is not None:
        if isinstance(region_of_interest, str):
            region_of_interest = load_glacier_outlines(region_of_interest)
        if buffer_region_by is not None:
            region_of_interest = gis.buffer_4326_shp(region_of_interest, buffer_region_by)
        region_of_interest = gis.simplify_4326_shp(region_of_interest)
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
    elif product == "glaciers":
        product = "G"
    else:
        raise ValueError(f"Argument product should be either glaciers or complexes not \"{product}\".")
    rgi_files = os.listdir(rgi_path)
    for file in rgi_files:
        if re.match(f"RGI2000-v7\\.0-{product}-{o1code[:2]}_.*", file):
            file_path = os.path.join(rgi_path, file)
            if file[-8:] == ".feather":
                o1region = gpd.read_feather(file_path)
            elif file[-4:] == ".shp" or os.path.isdir(file_path):
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
            warnings.warn(f"Dropping {sum(small_glacier_mask)} glaciers < 1 km².")
        o1region = o1region[~small_glacier_mask]
    return o1region


def load_o2region(o2code: str, product: str = "complexes") -> gpd.GeoDataFrame:
    o1region = load_o1region(o2code[:2], product)
    return o1region[o1region["o2region"]==o2code[:5]]


def load_basins(rgi_ids: list[str]) -> gpd.GeoDataFrame:
    if len(rgi_ids) > 1:
        assert(all([id[:17]==rgi_ids[0][:17]] for id in rgi_ids))
    rgi_o1_gpdf = load_o1region(rgi_ids[0].split("-")[3], product="glaciers")
    id_to_index_series = pd.Series(data=rgi_o1_gpdf.index, index=rgi_o1_gpdf.rgi_id)
    return rgi_o1_gpdf.loc[id_to_index_series.loc[rgi_ids].values]


def load_glacier_outlines(identifier: str|list[str]) -> shapely.MultiPolygon:
    if isinstance(identifier, list):
        out = load_basins(identifier)
    elif len(identifier) == (7+4+1+2+5+4) and identifier.split("-")[:3] == ["RGI2000", "v4.1", "G"]:
        out = load_basins([identifier])
    # the pattern is rather allowing, set it to "^(-?[012][0-9]){2}(_[a-z]+){1,5}(_[0-9][a-z][0-9]?)?$" to make it tight
    elif len(identifier) >= 5 and re.match("^(-?[012][0-9]){2}$", identifier[:5]):
        out = load_o2region(identifier[:5])
    elif re.match("[012][0-9](_[a-z]+)+", identifier):
        out = load_o1region(identifier[:2])
    else:
        raise ValueError(f"Provided o1, o2, or RGI identifiers. \"{identifier}\" not understood.")
    return out.unary_union
__all__.append("load_glacier_outlines")


def nan_unique(data: np.typing.ArrayLike) -> list:
    return [element for element in np.unique(data) if not np.isnan(element)]
__all__.append("nan_unique")

__all__ = sorted(__all__)
