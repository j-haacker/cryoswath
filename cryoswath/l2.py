import geopandas as gpd
import h5py
import itertools
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from pyarrow.lib import ArrowInvalid
from pyproj import CRS
import re
import shapely
import shutil
from tables import NaturalNameWarning
# from threading import Event, Thread
import warnings

from .misc import *
from . import l1b

__all__ = list()


def from_id(track_idx: pd.DatetimeIndex|str, *,
            reprocess: bool|pd.Timestamp = True,
            save_or_return: str = "both",
            cache_fullname: str = None,
            cores: int = len(os.sched_getaffinity(0)),
            **kwargs) -> tuple[gpd.GeoDataFrame]:
    # this function collects processed data and processes the remaining.
    # combining new and old data can show unexpected behavior if
    # settings changed.
    # if you have the data cached by providing cache_fullname, make sure to
    # fix CRS by passing `crs=xxxx`; otherwise crs is determined per track.
    # tbi: throw error if crs not passed.
    # if ESA complains there were too many parallel ftp connections, reduce
    # the number of cores. 8 cores worked for me, 16 was too many
    if not isinstance(track_idx, pd.DatetimeIndex):
        track_idx = pd.DatetimeIndex(track_idx if isinstance(track_idx, list) else [track_idx])
    track_idx = track_idx.sort_values()
    if track_idx.tz != None:
        track_idx = track_idx.tz_localize(None)
    if isinstance(reprocess, str):
        reprocess = pd.Timestamp(reprocess)
    # somehow the download thread prevents the processing of tracks. it may
    # be due to GIL lock. for now, it is just disabled, so one has to
    # download in advance. on the fly is always possible, however, with
    # parallel processing this can lead to issues because ESA blocks ftp
    # connections if there are too many.
    print("[note] You can speed up processing substantially by previously downloading the L1b data.")
    # stop_event = Event()
    # download_thread = Thread(target=l1b.download_wrapper,
    #                          kwargs=dict(track_idx=track_idx, num_processes=8, stop_event=stop_event),
    #                          name="dl_l1b_mother_thread",
    #                          daemon=False)
    # download_thread.start()
    try:
        start_datetime, end_datetime = track_idx.sort_values()[[0,-1]]
        # ! below will not return data that is cached, even if save_or_return="both"
        # this is a flaw in the current logic. rework.
        if not reprocess and cache_fullname is not None and save_or_return != "return":
            if os.path.isfile(cache_fullname):
                present_months = []
                def collect_present_months(name, node):
                    nonlocal present_months
                    if isinstance(node, h5py.Dataset) or not name.split("/")[-1].startswith("t_"):
                        return None
                    present_months.append(pd.to_datetime(name.split("/")[-1], format="t_%Y-%m"))
                with h5py.File(cache_fullname, "r") as h5:
                    h5.visititems(collect_present_months)
                skip_months = pd.DatetimeIndex(present_months).unique().sort_values()
                print(skip_months)
            else:
                skip_months = []
        swath_list = []
        poca_list = []
        kwargs["cs_full_file_names"] = load_cs_full_file_names(update="no")
        for current_month in pd.date_range(start_datetime.normalize()-pd.DateOffset(day=1),
                                           end_datetime, freq="MS"):
            if not reprocess\
                    and cache_fullname is not None\
                    and save_or_return != "return"\
                    and current_month.tz_localize(None) in skip_months:
                print("Skipping cached month", current_month.strftime("%Y-%m"))
                continue
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
            # # indices per month with work-around :/ should be easier
            # current_track_indices = pd.Series(index=track_idx).loc[current_month:current_month+pd.offsets.MonthBegin(1)].index
            current_track_indices = track_idx[np.logical_and(track_idx>=current_month,
                                                             track_idx<current_month+pd.offsets.MonthBegin(1))]
            if len(current_track_indices) == 0:
                warnings.warn(f"No tracks to load data from for month {current_month}.")
                continue
            collective_swath_poca_list = []
            for batch in itertools.batched(current_track_indices, cores*3):
                if cores > 1 and len(batch) > 1:
                    with Pool(processes=cores) as p:
                        # function is defined at the bottom of this module
                        collective_swath_poca_list.extend(p.starmap(process_track,
                            [(idx, reprocess, l2_paths, ["return" if save_or_return == "return" else "both"],
                            current_subdir, kwargs) for idx in batch], chunksize=1))
                else:
                    for idx in batch:
                        collective_swath_poca_list.append(
                            process_track(idx, reprocess, l2_paths, ["return" if save_or_return == "return" else "both"],
                                        current_subdir, kwargs))
            # ensure all data within `max_elev_diff`, in same CRS, and clip to bbox
            # note: if no crs is provided, check will not be done. if CRS mismatch,
            #       error occurs later
            if any(key in kwargs for key in ["max_elev_diff", "crs", "bbox"]):
                for i in range(len(collective_swath_poca_list)):
                    tmp = list(collective_swath_poca_list[i])
                    for j in range(len(tmp)):
                        if not tmp[j].empty:
                            if "max_elev_diff" in kwargs:
                                tmp[j] = limit_filter(tmp[j], "h_diff", kwargs["max_elev_diff"])
                            if "crs" in kwargs:
                                tmp[j] = tmp[j].to_crs(kwargs["crs"])
                            # bbox must be `shapely.(Multi)Polygon` in same crs
                            if "bbox" in kwargs:
                                tmp[j] = tmp[j].clip(kwargs["bbox"])
                    collective_swath_poca_list[i] = tuple(tmp)
            if cache_fullname is not None:
                # when postprocessing, loading the data caching here takes a substatial
                # amount of time. not sure, but maybe the format can be improved. there
                # is parquet or the data could be saved per month using
                # to_hdf(format="fixed")
                
                # in the next step, the data will be cached on disk for later use. this bears
                # the risk of introducing data gaps that are very difficult to detect.
                # the main rationale is: be sure not to damage the data and be careful
                # not to loose the cache.
                # this implementation balances the chance loosing the current data with
                # the cost of copying by backing up once per hour
                if os.path.isfile(cache_fullname):
                    if os.path.isfile(cache_fullname+"__backup"):
                        # if backup older than 1 hour, renew
                        if (pd.Timestamp.now()-pd.to_datetime(os.stat(cache_fullname+"__backup").st_mtime, unit="s"))\
                                > pd.to_timedelta(1, unit="h"):
                            os.remove(cache_fullname+"__backup")
                            shutil.copyfile(cache_fullname, cache_fullname+"__backup")
                    # if no backup, make one
                    else:
                        shutil.copyfile(cache_fullname, cache_fullname+"__backup")
                for l2_type, i in zip(["swath", "poca"], [0, 1]):
                    l2_data = pd.concat([item[i] for item in collective_swath_poca_list])
                    if l2_data.empty:
                        warnings.warn(f"No {l2_type} data at all for month {current_month}.")
                        continue
                    if l2_type == "swath":
                        l2_data.index = l2_data.index.get_level_values(0).astype(np.int64) \
                                        + l2_data.index.get_level_values(1)
                    else:
                        l2_data.index = l2_data.index.astype(np.int64)
                    l2_data.rename_axis("time", inplace=True)
                    l2_data.sort_index(inplace=True)
                    chunks = xycut(l2_data)
                    
                    # now, try to add current data to cache. if anything happens, restore
                    # backup to ensure integrity
                    try:
                        for chunk in chunks:
                            tmp = pd.DataFrame(index=chunk["data"].index, 
                                            data=pd.concat([chunk["data"].h_diff,
                                                            chunk["data"].geometry.get_coordinates()],
                                                            axis=1, copy=False))
                            # below, x and y will be stored as integers which rounds them
                            # implicitly. this introduces a logic-flaw in the chunking, where
                            # floor-division is used. this makes an extra step necessary. note that
                            # pandas does not offer `.floor()`. w/out testing, I assume
                            # `round(x-.5)` to perform better than x//1.
                            tmp[["x", "y"]] = (tmp[["x", "y"]]-.5).round()
                            # below hides warnings about a minus sign in node names. this can safely be ignored.
                            warnings.filterwarnings('ignore', category=NaturalNameWarning)
                            tmp.astype(dict(h_diff=np.float32, x=np.int32, y=np.int32))\
                            .to_hdf(cache_fullname, key="/".join(
                                [l2_type, f'x_{chunk["x_interval_start"]:.0f}_{chunk["x_interval_stop"]:.0f}',
                                          f'y_{chunk["y_interval_start"]:.0f}_{chunk["y_interval_stop"]:.0f}',
                                          current_month.strftime("t_%Y-%m")]
                                    ), mode="a", format="fixed")
                            warnings.filterwarnings('default', category=NaturalNameWarning)
                    except:
                        if not os.path.isfile(cache_fullname):
                            raise
                        os.remove(cache_fullname)
                        warnings.warn(
                            "There was an error while caching l2 data. Since this can lead to "
                            + "missing data, the cache has been removed and it was attempted to "
                            + "restore a backup. If this was successful, there should NOT be a "
                            + "__backup file in data/tmp. If there still is, please decide how to "
                            + "proceed yourself (you could remove all temporary files and have "
                            + "them produced freshly).")
                        shutil.move(cache_fullname+"__backup", cache_fullname)
                        raise
                    # tidying up the backup was moved to parent process
                    # else:
                    #     if os.path.isfile(cache+"__backup"):
                    #         os.remove(cache+"__backup")
            if save_or_return != "save":
                    swath_list.append(pd.concat([item[0] for item in collective_swath_poca_list]))
                    poca_list.append(pd.concat([item[1] for item in collective_swath_poca_list]))
            del collective_swath_poca_list
            print("done processing", current_month)
        if save_or_return != "save":
            if swath_list == []:
                swath_list = pd.DataFrame()
            else:
                swath_list = pd.concat(swath_list)
            if poca_list == []:
                poca_list = pd.DataFrame()
            else:
                poca_list = pd.concat(poca_list)
            return swath_list, poca_list
    except:
        # print("Waiting for download threads to join.")
        # stop_event.set()
        # print("Waiting for download threads to join.")
        # download_thread.join(30)
        raise
__all__.append("from_id")


def from_processed_l1b(l1b_data: l1b.l1b_data = None,
                       max_elev_diff: float = None,
                       **gdf_kwargs) -> gpd.GeoDataFrame:
    # kwargs: crs, max_elev_diff, input to GeoDataFrame
    if l1b_data is None:
        print("No data passed.")
        # ! empty GeoDataFrames can't have a CRS
        # super().__init__(crs=crs, **kwargs)
        return gpd.GeoDataFrame()
    tmp = l1b_data.to_dataframe().dropna(axis=0, how="any")
    if max_elev_diff is not None and "h_diff" in tmp:
        tmp = limit_filter(tmp, "h_diff", max_elev_diff)
        if tmp.empty:
            return gpd.GeoDataFrame()
    if isinstance(tmp.index, pd.MultiIndex): #
        tmp.rename_axis(("time", "sample"), inplace=True)
        tmp.index = tmp.index.set_levels(pd.DatetimeIndex(tmp["time"].groupby(level=0).first(), tz="UTC"), level=0)
    elif tmp.index.name[:4].lower()=="time":
        tmp.rename_axis(("time"), inplace=True)
        tmp.index = pd.DatetimeIndex(tmp["time"], tz="UTC")
    elif tmp.index.name[:2].lower()=="ns":
        tmp.rename_axis(("sample"), inplace=True)
    else:
        warnings.warn("Unexpected index name. May lead to issues.")
    tmp.drop(columns=[col for col in ["time", "sample"] if col in tmp.columns], inplace=True)
    # convert either lat, lon or x, y data to points assuming that any crs but 4326 uses x, y coordinates
    if l1b_data.CRS == CRS.from_epsg(4326):
        geometry = gpd.points_from_xy(tmp.lon, tmp.lat)
        tmp.drop(columns=["lat", "lon"], inplace=True)
    else:
        geometry = gpd.points_from_xy(tmp.x, tmp.y)
        tmp.drop(columns=["x", "y"], inplace=True)
    res = gpd.GeoDataFrame(tmp, geometry=geometry, crs=l1b_data.CRS, **filter_kwargs(gpd.GeoDataFrame, gdf_kwargs, blacklist=["crs"]))
    if "crs" in gdf_kwargs:
        res = res.to_crs(gdf_kwargs.pop("crs"))
    return res
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
    res = data[np.abs(data[column])<limit]
    # if a multiindex is used, it retains the level values for the deleted
    # rows. this is, I believe, not the expected result of a filter.
    if isinstance(res.index, pd.MultiIndex):
        res.index = res.index.remove_unused_levels()
    return res
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
def process_track(idx, reprocess, l2_paths, save_or_return, current_subdir, kwargs):
    print("getting", idx)
    # print("kwargs", wargs)
    try:
        if any(l2_paths.loc[idx,:].isnull()) or (isinstance(reprocess, bool) and reprocess):
            raise FileNotFoundError()
        if isinstance(reprocess, pd.Timestamp):
            # print(reprocess, pd.Timestamp(
            #     os.stat(os.path.join(l2_swath_path, current_subdir, l2_paths.loc[idx, "swath"])).st_mtime,
            #     unit="s"))
            assert(reprocess<pd.Timestamp(
                os.stat(os.path.join(l2_swath_path, current_subdir, l2_paths.loc[idx, "swath"])).st_mtime,
                unit="s"))
            assert(reprocess<pd.Timestamp(
                os.stat(os.path.join(l2_swath_path, current_subdir, l2_paths.loc[idx, "poca"])).st_mtime,
                unit="s"))
        if save_or_return != "save":
            swath_poca_tuple = (
                gpd.read_feather(os.path.join(l2_swath_path, current_subdir,
                                                l2_paths.loc[idx, "swath"])),
                gpd.read_feather(os.path.join(l2_poca_path, current_subdir,
                                                l2_paths.loc[idx, "poca"])))
    except (KeyError, FileNotFoundError, AssertionError, ArrowInvalid):
        if "cs_full_file_names" not in locals():
            if "cs_full_file_names" in kwargs:
                cs_full_file_names = kwargs.pop("cs_full_file_names")
            else:
                cs_full_file_names = load_cs_full_file_names(update="no")
        l1b_kwargs = filter_kwargs(l1b.l1b_data, kwargs)
        to_l2_kwargs = filter_kwargs(l1b.l1b_data.to_l2, kwargs, blacklist=["swath_or_poca"],
                                     whitelist=["crs", "max_elev_diff"])
        try:
            tmp = l1b.l1b_data.from_id(cs_time_to_id(idx), **l1b_kwargs)
            swath_poca_tuple = tmp.to_l2(swath_or_poca="both", **to_l2_kwargs)
            tmp.close()
        except Exception as err:
            if isinstance(err, KeyboardInterrupt):
                raise
            else:
                # raise
                warnings.warn(f"Error {str(err)} occured while processing l1b {idx}. Continuing with next file.")
                swath_poca_tuple = (gpd.GeoDataFrame(), gpd.GeoDataFrame())
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
        
