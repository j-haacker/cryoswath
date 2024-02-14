
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


def from_id(track_idx: pd.DatetimeIndex|str,
            max_elev_diff: float = 100):
    # this function collects processed data and processes the remaining.
    # combining new and old data can show unexpected behavior if
    # settings changed.
    if not isinstance(track_idx, pd.DatetimeIndex):
        track_idx = pd.DatetimeIndex(track_idx if isinstance(track_idx, list) else [track_idx])
    if track_idx.tz == None: track_idx.tz_localize("UTC")
    start_datetime, end_datetime = track_idx.sort_values()[[0,-1]]
    l2_list = []
    for current_month in pd.date_range(start_datetime-pd.offsets.MonthBegin(1, normalize=True), end_datetime, freq="M"):
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
            if current_track_idx in l2_paths.index:
                l2_list.append(gpd.read_feather(os.path.join(current_L2_base_path, l2_paths.loc[current_track_idx])))
            else:
                # print("processing", track_idx)
                l2_buffer = limit_filter(l1b.l1b_data.from_id(cs_time_to_id(current_track_idx)).to_l2(),
                                         "h_diff", max_elev_diff)
                l2_list.append(l2_buffer)
                # save results. make optional?
                if "cs_full_file_names" not in locals():
                    cs_full_file_names = load_cs_full_file_names()
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
        # "flatten" data to avoid dim order confusion and convert to dataframe
        l1b_data = l1b_data.to_dataframe().dropna(axis=0, how="any").rename_axis(("time", "sample"))
        l1b_data.index = l1b_data.index.set_levels(pd.DatetimeIndex(l1b_data["time"].groupby(level=0).first(), tz="UTC"), level=0)
        l1b_data.drop(columns="time", inplace=True)
        # convert either lat, lon or x, y data to points assuming that any crs but 4326 uses x, y coordinates
        if crs == CRS.from_epsg(4326):
            geometry = gpd.points_from_xy(l1b_data.lon, l1b_data.lat)
            l1b_data.drop(columns=["lat", "lon"], inplace=True)
        else:
            geometry = gpd.points_from_xy(l1b_data.x, l1b_data.y)
            l1b_data.drop(columns=["x", "y"], inplace=True)
        return gpd.GeoDataFrame(l1b_data, geometry=geometry, crs=crs, **kwargs)
__all__.append("from_processed_l1b")


def limit_filter(data: pd.DataFrame, column: str, limit: float):
    return data[np.abs(data[column])<limit]
__all__.append("limit_filter")


def process_and_save(region_of_interest: str|shapely.Polygon,
                     start_datetime: pd.Timestamp,
                     end_datetime: pd.Timestamp):
    cs_tracks = gis.load_cs_ground_tracks().loc[start_datetime:end_datetime]
    # find all tracks that intersect the buffered region of interest.
    # mind that this are calculations on a sphere. currently, the
    # polygon is transformed to ellipsoidial coordinates. not a 100 %
    # sure that this doesn't raise issues close to the poles.
    
    if not isinstance(region_of_interest, shapely.Polygon):
        if not re.match("[012][0-9]-[012][0-9]", region_of_interest):
            raise Exception("Error: can only parse RGI o2 codes, e.g., 19-15.")
        region_of_interest = load_o2region(region_of_interest).unary_union
    cs_tracks = cs_tracks[cs_tracks.intersects(gis.buffer_4326_shp(region_of_interest, 30000))]
    print(cs_tracks.shape[0], "tracks remain")
    from_id(cs_tracks.index)
    print("processing L1b -> L2 finished")
    return 0
__all__.append("process_and_save")
