import geopandas as gpd
import inspect
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from pyproj import CRS
import re
import shapely
from threading import Event, Thread
import warnings

from .misc import *
from . import gis, l1b

__all__ = list()


def from_id(track_idx: pd.DatetimeIndex|str, *,
            reprocess: bool = True,
            save_or_return: str = "both",
            cores: int = len(os.sched_getaffinity(0)),
            **kwargs) -> tuple[gpd.GeoDataFrame]:
    # this function collects processed data and processes the remaining.
    # combining new and old data can show unexpected behavior if
    # settings changed.
    # if ESA complains there were too many parallel ftp connections, reduce
    # the number of cores. 8 cores worked for me, 16 was too many
    if not isinstance(track_idx, pd.DatetimeIndex):
        track_idx = pd.DatetimeIndex(track_idx if isinstance(track_idx, list) else [track_idx])
    if track_idx.tz == None:
        track_idx.tz_localize("UTC")
    # somehow the download thread prevents the processing of tracks. it may
    # be due to GIL lock. for now, it is just disabled, so one has to
    # download in advance. on the fly is always possible, however, with
    # parallel processing this can lead to issues because ESA blocks ftp
    # connections if there are too many.
    print("Note that you can speed up processing substantially by previously downloading the L1b data.")
    # stop_event = Event()
    # download_thread = Thread(target=l1b.download_wrapper,
    #                          kwargs=dict(track_idx=track_idx, num_processes=8, stop_event=stop_event),
    #                          name="dl_l1b_mother_thread",
    #                          daemon=False)
    # download_thread.start()
    try:
        start_datetime, end_datetime = track_idx.sort_values()[[0,-1]]
        swath_list = []
        poca_list = []
        kwargs["cs_full_file_names"] = load_cs_full_file_names(update="no")
        for current_month in pd.date_range(start_datetime.normalize()-pd.offsets.MonthBegin(), end_datetime, freq="MS"):
            current_subdir = current_month.strftime(f"%Y{os.path.sep}%m")
            l2_paths = pd.DataFrame(columns=["swath", "poca"])
            for l2_type in ["swath", "poca"]:
                if os.path.isdir(os.path.join(data_path, f"L2_{l2_type}", current_subdir)):
                    for filename in os.listdir(os.path.join(data_path, f"L2_{l2_type}", current_subdir)):
                        match = re.search(cryosat_id_pattern, filename)
                        if match is not None:
                            l2_paths.loc[cs_id_to_time(match.group()), l2_type] = filename
                else:
                    os.makedirs(os.path.join(data_path, f"L2_{l2_type}", current_subdir))
            print("start processing", current_month)
            with Pool(processes=cores) as p:
                # function is defined at the bottom of this module
                collective_swath_poca_list = p.starmap(
                    process_track,
                    [(idx, reprocess, l2_paths, save_or_return, data_path, current_subdir, kwargs)
                     for idx
                     # indices per month with work-around :/ should be easier
                     in pd.Series(index=track_idx).loc[current_month:current_month+pd.offsets.MonthBegin(1)].index],
                     chunksize=1)
            if save_or_return != "save":
                for swath_poca_tuple in collective_swath_poca_list: # .get()
                    swath_list.append(swath_poca_tuple[0])
                    poca_list.append(swath_poca_tuple[1])
            print("done processing", current_month)
        if save_or_return != "save":
            return pd.concat(swath_list), pd.concat(poca_list)
    except:
        # print("Waiting for download threads to join.")
        # stop_event.set()
        # print("Waiting for download threads to join.")
        # download_thread.join(30)
        raise
__all__.append("from_id")


def from_processed_l1b(l1b_data: l1b.l1b_data = None,
                       **kwargs) -> gpd.GeoDataFrame:
    # kwargs: crs, max_elev_diff, input to GeoDataFrame
    if l1b_data is None:
        print("No data passed.")
        # ! empty GeoDataFrames can't have a CRS
        # super().__init__(crs=crs, **kwargs)
        pass
    else:
        if "crs" in kwargs:
            crs = gis.ensure_pyproj_crs(kwargs.pop("crs"))
        else:
            crs = CRS.from_epsg(3413)
        l1b_data = l1b_data.to_dataframe().dropna(axis=0, how="any")
        if "max_elev_diff" in kwargs and "h_diff" in l1b_data:
            l1b_data = limit_filter(l1b_data, "h_diff", kwargs.pop("max_elev_diff"))
        if isinstance(l1b_data.index, pd.MultiIndex): #
            l1b_data.rename_axis(("time", "sample"), inplace=True)
            l1b_data.index = l1b_data.index.set_levels(pd.DatetimeIndex(l1b_data["time"].groupby(level=0).first(), tz="UTC"), level=0)
        elif l1b_data.index.name[:4].lower()=="time":
            l1b_data.rename_axis(("time"), inplace=True)
            l1b_data.index = pd.DatetimeIndex(l1b_data["time"], tz="UTC")
        elif l1b_data.index.name[:2].lower()=="ns":
            l1b_data.rename_axis(("sample"), inplace=True)
        else:
            warnings.warn("Unexpected index name. May lead to issues.")
        l1b_data.drop(columns=[col for col in ["time", "sample"] if col in l1b_data.columns], inplace=True)
        # convert either lat, lon or x, y data to points assuming that any crs but 4326 uses x, y coordinates
        if crs == CRS.from_epsg(4326):
            geometry = gpd.points_from_xy(l1b_data.lon, l1b_data.lat)
            l1b_data.drop(columns=["lat", "lon"], inplace=True)
        else:
            geometry = gpd.points_from_xy(l1b_data.x, l1b_data.y)
            l1b_data.drop(columns=["x", "y"], inplace=True)
        return gpd.GeoDataFrame(l1b_data, geometry=geometry, crs=crs, **kwargs)
__all__.append("from_processed_l1b")


def grid(l2_data: gpd.GeoDataFrame, spatial_res_meter: float, aggregation_function: callable) -> pd.DataFrame:
    # define how to grid
    def cell_bounds(number: float):
        floor = np.floor(number/spatial_res_meter)*spatial_res_meter
        return slice(floor, floor+spatial_res_meter)
    # split data into chunks
    # reason: it seemed that index accessing time increases much for
    # large data sets. remember it is not a database index (pandas
    # doesn't whether it is sorted)
    n_split = int((l2_data.shape[0]/.5e6)**.5)
    minx, miny, maxx, maxy = l2_data.total_bounds
    delx = (maxx-minx)//n_split+1 # + 1 m to be sure to cover the edges, probably not necessary
    dely = (maxy-miny)//n_split+1
    l2_list = [l2_data.cx[x:x+delx,y:y+dely] for x in np.arange(minx, maxx, delx) for y in np.arange(miny, maxy, dely)]
    del l2_data
    l2_list = [df for df in l2_list if not df.empty]
    # here is probably improvement potential. e.g. assign x, y, t
    # indices and use groupby. the below used many function calls.
    # further, appending to a list needs to allocate memory per grid
    # cell.
    gridded_list = []
    for parent_cell in l2_list:
        while not parent_cell.empty:
            print("# point data:", parent_cell.shape[0])
            loc = parent_cell.iloc[0].geometry
            # print("location", loc)
            x_slice = cell_bounds(loc.x)
            y_slice = cell_bounds(loc.y)
            subset = parent_cell.cx[x_slice,y_slice]["h_diff"]
            result = aggregation_function(subset)
            parent_cell.drop(index=subset.index, inplace=True)
            print("# subset data:", subset.shape[0])
            gridded_list.append(pd.concat([result], keys=[(x_slice.start,y_slice.start)], names=["x", "y", "time"]))
        # consider saving a backup to disk after each parent_cell
    return pd.concat(gridded_list)


def limit_filter(data: pd.DataFrame, column: str, limit: float) -> pd.DataFrame:
    if np.isnan(limit) or limit <= 0:
        return data
    return data[np.abs(data[column])<limit]
__all__.append("limit_filter")


def process_and_save(region_of_interest: str|shapely.Polygon,
                     start_datetime: str|pd.Timestamp,
                     end_datetime: str|pd.Timestamp, *,
                     buffer_region_by: float = 10_000,
                     **kwargs):
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    cs_tracks = load_cs_ground_tracks(region_of_interest, start_datetime, end_datetime, buffer_region_by=buffer_region_by)
    from_id(cs_tracks.index, save_or_return="save", **kwargs)
    print("processing L1b -> L2 finished")
    return 0
__all__.append("process_and_save")

__all__ = sorted(__all__)


# local helper function. can't be defined where it is needed because of namespace issues
def process_track(idx, reprocess, l2_paths, save_or_return, data_path, current_subdir, kwargs):
    print("getting", idx)
    # print("kwargs", wargs)
    try:
        if reprocess or any(l2_paths.loc[idx,:].isnull()):
            raise FileNotFoundError()
        if save_or_return != "save":
            swath_poca_tuple = (
                gpd.read_feather(os.path.join(l2_swath_path, current_subdir,
                                                l2_paths.loc[idx, "swath"])),
                gpd.read_feather(os.path.join(l2_poca_path, current_subdir,
                                                l2_paths.loc[idx, "poca"])))
    except (KeyError, FileNotFoundError):
        if "cs_full_file_names" not in locals():
            if "cs_full_file_names" in kwargs:
                cs_full_file_names = kwargs.pop("cs_full_file_names")
            else:
                cs_full_file_names = load_cs_full_file_names(update="no")
        # filter l1b_data kwargs
        params = inspect.signature(l1b.l1b_data).parameters
        l1b_kwargs = {k: v for k, v in kwargs.items() if k in params}
        # filter to_l2 kwargs
        params = inspect.signature(l1b.l1b_data.to_l2).parameters
        to_l2_kwargs = {k: v for k, v in kwargs.items() if k in params and k != "swath_or_poca"}
        swath_poca_tuple = l1b.l1b_data.from_id(cs_time_to_id(idx), **l1b_kwargs)\
                                       .to_l2(swath_or_poca="both", **to_l2_kwargs)
        if save_or_return != "return":
            print("saving", idx)
            # ! consider writing empty files
            # the below skips if there are no data. this means, that processing is
            # attempted the next time again. I consider this safer and the
            # performance loss is on the order of seconds. however, there might be
            # better options
            try:
                swath_poca_tuple[0].to_feather(os.path.join(l2_swath_path, current_subdir,
                                                            cs_full_file_names.loc[idx]+".feather"))
                swath_poca_tuple[1].to_feather(os.path.join(l2_poca_path, current_subdir,
                                                            cs_full_file_names.loc[idx]+".feather"))
            except ValueError:
                if swath_poca_tuple[0].empty:
                    which = "Neither POCA nor swath"
                else:
                    which = "No POCA"
                warnings.warn(f"{which} points for {cs_time_to_id(idx)}.")
    if save_or_return != "save":
        return swath_poca_tuple
    else: # not sure that its necessary
        return 0
        
