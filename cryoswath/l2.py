import fnmatch
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import CRS
import re
import shapely
import warnings

from .misc import *
from . import gis, l1b

__all__ = list()


def from_id(track_idx: pd.DatetimeIndex|str, *,
            reprocess: bool = True,
            save_or_return: str = "both",
            **kwargs) -> tuple[gpd.GeoDataFrame]:
    # this function collects processed data and processes the remaining.
    # combining new and old data can show unexpected behavior if
    # settings changed.
    if not isinstance(track_idx, pd.DatetimeIndex):
        track_idx = pd.DatetimeIndex(track_idx if isinstance(track_idx, list) else [track_idx])
    if track_idx.tz == None: track_idx.tz_localize("UTC")
    start_datetime, end_datetime = track_idx.sort_values()[[0,-1]]
    swath_list = []
    poca_list = []
    for current_month in pd.date_range(start_datetime.normalize()-pd.offsets.MonthBegin(), end_datetime,freq="MS"):
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
        # below could be parallelized in different ways:
        # a) downloading missing L1b; b) processing L1b; c) collecting
        for current_track_idx in pd.Series(index=track_idx).loc[current_month:current_month+pd.offsets.MonthBegin(1)].index: # super work-around :/
            print("getting", current_track_idx)
            try:
                if reprocess or any(l2_paths.loc[current_track_idx,:].isnull()):
                    raise FileNotFoundError()
                if save_or_return != "save":
                    swath_list.append(gpd.read_feather(
                        os.path.join(data_path, "L2_swath", current_subdir, l2_paths.loc[current_track_idx, "swath"])))
                    poca_list.append(gpd.read_feather(
                        os.path.join(data_path, "L2_poca", current_subdir, l2_paths.loc[current_track_idx, "poca"])))
            except (KeyError, FileNotFoundError):
                # print("processing", current_track_idx)
                # save results. make optional?
                if "cs_full_file_names" not in locals():
                    cs_full_file_names = load_cs_full_file_names(update="no")
                if save_or_return != "save":
                    swath_list.append(l2_buffer[0])
                    poca_list.append(l2_buffer[1])
                if save_or_return != "return":
                    # ! consider writing empty files
                    # the below skips if there are no data. this means, that processing is
                    # attempted the next time again. I consider this safer and the
                    # performance loss is on the order of seconds. however, there might be
                    # better options
                    try:
                        l2_buffer[0].to_feather(os.path.join(data_path, "L2_swath", current_subdir,
                                                             cs_full_file_names.loc[current_track_idx]+".feather"))
                        l2_buffer[1].to_feather(os.path.join(data_path, "L2_poca", current_subdir,
                                                             cs_full_file_names.loc[current_track_idx]+".feather"))
                    except ValueError:
                        if l2_buffer[0].empty:
                            which = "Neither POCA nor swath"
                        else:
                            which = "No POCA"
                        warnings.warn(f"{which} points for {cs_time_to_id(current_track_idx)}.")
                # print("done processing")
    if save_or_return != "save":
        return pd.concat(swath_list), pd.concat(poca_list)
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
    from_id(cs_tracks.index, **kwargs)
    print("processing L1b -> L2 finished")
    return 0
__all__.append("process_and_save")
