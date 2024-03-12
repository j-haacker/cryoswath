
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import CRS
import re
import shapely

from .misc import *
from . import gis, l1b

__all__ = list()


def from_id(track_idx: pd.DatetimeIndex|str, *,
            max_elev_diff: float = np.nan,
            reprocess: bool = True):
    # this function collects processed data and processes the remaining.
    # combining new and old data can show unexpected behavior if
    # settings changed.
    if not isinstance(track_idx, pd.DatetimeIndex):
        track_idx = pd.DatetimeIndex(track_idx if isinstance(track_idx, list) else [track_idx])
    if track_idx.tz == None: track_idx.tz_localize("UTC")
    start_datetime, end_datetime = track_idx.sort_values()[[0,-1]]
    l2_list = []
    for current_month in pd.date_range(start_datetime.normalize()-pd.offsets.MonthBegin(), end_datetime,freq="MS"):
        current_L2_base_path = os.path.join("..", "data", "L2", current_month.strftime(f"%Y{os.path.sep}%m"))
        if os.path.isdir(current_L2_base_path):
            l2_paths = os.listdir(current_L2_base_path)
            l2_paths = pd.Series(l2_paths, name="l2_paths")
            l2_paths.index = pd.DatetimeIndex([pd.to_datetime(l2_file[19:34]) for l2_file in l2_paths])
        else:
            l2_paths = pd.Series()
            os.makedirs(current_L2_base_path)
        # below could be parallelized in different ways:
        # a) downloading missing L1b; b) processing L1b; c) collecting
        for current_track_idx in pd.Series(index=track_idx).loc[current_month:current_month+pd.offsets.MonthBegin(1)].index: # super work-around :/
            print("appending", current_track_idx)
            if not reprocess and current_track_idx in l2_paths.index:
                l2_list.append(gpd.read_feather(os.path.join(current_L2_base_path, l2_paths.loc[current_track_idx])))
            else:
                # print("processing", track_idx)
                l2_buffer = limit_filter(l1b.l1b_data.from_id(cs_time_to_id(current_track_idx)).to_l2(tidy=False),
                                         "h_diff", max_elev_diff)
                l2_list.append(l2_buffer)
                # save results. make optional?
                if "cs_full_file_names" not in locals():
                    cs_full_file_names = load_cs_full_file_names(update="no")
                l2_buffer.to_feather(os.path.join(current_L2_base_path, cs_full_file_names.loc[current_track_idx]+".feather"))
                # print("done processing")
    return pd.concat(l2_list)
__all__.append("from_id")


def from_processed_l1b(l1b_data: l1b.l1b_data|None = None, crs: CRS = CRS.from_epsg(3413), **kwargs) -> gpd.GeoDataFrame:
    # print(data)
    if l1b_data is None:
        print("No data passed.")
        # ! empty GeoDataFrames can't have a CRS
        # super().__init__(crs=crs, **kwargs)
        pass
    else:
        if isinstance(crs, int): crs = CRS.from_epsg(crs)
        l1b_data = l1b_data.to_dataframe().dropna(axis=0, how="any")
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
