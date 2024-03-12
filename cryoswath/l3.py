from dateutil.relativedelta import relativedelta
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import re
import shapely
import xarray as xr

from . import gis
from . import l2
from .misc import *

__all__ = list()

class l3_data(xr.Dataset):
    def __init__(self, rgi_long_code, spatial_res_meter):
        # it is difficult to find gridding conventions. here, for now it
        # is decided to use the coordinates of the bottom left cell
        # corner because xarray uses pcolormesh for plotting which
        # interprets the coordinates this way.

        # use global dataset instead?
        rgi_o2_gpdf = gpd.read_file(f"../data/auxiliary/RGI/RGI2000-v7.0-o2regions")
        # ! only Arctic
        region_bounds = rgi_o2_gpdf[rgi_o2_gpdf["long_code"]==rgi_long_code].geometry.to_crs(3413).bounds
        super().__init__(coords=dict(x=np.arange(region_bounds["minx"]//spatial_res_meter*spatial_res_meter,
                                                 region_bounds["maxx"], spatial_res_meter),
                                     y=np.arange(region_bounds["miny"]//spatial_res_meter*spatial_res_meter,
                                                 region_bounds["maxy"], spatial_res_meter)))

    # @classmethod
    # def from_l2(cls, l2_data:gpd.GeoDataFrame,
    #             agg_time: pd.DateOffset = pd.DateOffset(months=3),
    #             spatial_res_meter: float = 500,
    #             timestep: pd.DateOffset = pd.DateOffset(months=1)):
    #     # ! ensure that elevation difference to reference is present or offer options?
    #     rgi_o2_id = find_region_id(shapely.geometry.box(l2_data.total_bounds))
    #     l3_data = cls(rgi_o2_id, spatial_res_meter)
    #     def mad(data):
    #         return np.fabs(data - data.median()).median()
    #     for left in l3_data.x:
    #         for bottom in l3_data.y:
    #             # ! ensure index is timestamp
    #             l3_data.sel(x=left, y=bottom).elev_diff_median \
    #                 = l2_data.cx[left:left+spatial_res_meter,bottom:bottom+spatial_res_meter]["elev_diff"]\
    #                          .rolling(agg_time, center=True).median().to_numpy()
    #             l3_data.sel(x=left, y=bottom).elev_diff_mad \
    #                 = l2_data.cx[left:left+spatial_res_meter,bottom:bottom+spatial_res_meter]["elev_diff"]\
    #                          .rolling(agg_time, center=True).apply(mad, raw=True) # also compute mad


def build_dataset(region_of_interest: shapely.Polygon,
                  start_datetime: str|pd.Timestamp,
                  end_datetime: str|pd.Timestamp, *,
                  aggregation_period: relativedelta = relativedelta(months=3),
                  timestep: relativedelta = relativedelta(months=1),
                  spatial_res_meter: float = 500,
                  **kwargs):
    # if len(aggregation_period.kwds.keys()) != 1 \
    # or len(timestep.kwds.keys()) != 1 \
    # or list(aggregation_period.kwds.keys())[0] not in ["years", "months", "days"] \
    # or list(timestep.kwds.keys())[0] not in ["years", "months", "days"]:
    #     raise Exception("Only use one of years, months, days for agg_time and timestep.")
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)
    cs_tracks = gis.load_cs_ground_tracks()
    time_buffer = (aggregation_period-timestep)/2
    cs_tracks = cs_tracks.loc[start_datetime-time_buffer:end_datetime.normalize()+time_buffer+pd.offsets.Day(1)]
    if isinstance(region_of_interest, str) and re.match("[012][0-9]-[012][0-9]", region_of_interest):
        region_of_interest = load_o2region(region_of_interest).unary_union
    # find all tracks that intersect the buffered region of interest.
    # mind that this are calculations on a sphere. currently, the
    # polygon is transformed to ellipsoidal coordinates. not a 100 %
    # sure that this doesn't raise issues close to the poles.
    cs_tracks = cs_tracks[cs_tracks.intersects(gis.buffer_4326_shp(region_of_interest, 30000))]
    # I believe passing loading l2 data to the function prevents copying
    # on .drop. an alternative would be to define l2_data nonlocal
    # within the gridding function
    l3_data =  med_mad_cnt_grid(l2.from_id(cs_tracks.index), start_datetime=start_datetime, end_datetime=end_datetime,
                                aggregation_period=aggregation_period, timestep=timestep, spatial_res_meter=spatial_res_meter)
    l3_data.to_netcdf(build_path(region_of_interest, timestep, spatial_res_meter, aggregation_period))
    return l3_data
__all__.append("build_dataset")


def build_path(region_of_interest, timestep, spatial_res_meter, agg_time):
    region_id = find_region_id(region_of_interest)
    if list(timestep.kwds.values())[0]!=1:
        timestep_str = str(list(timestep.kwds.values())[0])+"-"
    else:
        timestep_str = ""
    timestep_str += list(timestep.kwds.keys())[0][:-1]+"ly"
    if spatial_res_meter == 1000 == 1000:
        spatial_res_str = "1km"
    elif np.floor(spatial_res_meter/1000) < 2:
        spatial_res_str = f"{spatial_res_meter}m"
    else:
        # if the trailing ".0" should be omitted, that needs to be implemented here
        spatial_res_str = f"{round(spatial_res_meter/1000, 1)}km"
    return os.path.join("..", "data", "L3", "_".join(
        [region_id, timestep_str, spatial_res_str+".nc"]))
__all__.append("build_path")
    

def med_mad_cnt_grid(l2_data: gpd.GeoDataFrame, *,
                     start_datetime: pd.Timestamp,
                     end_datetime: pd.Timestamp,
                     aggregation_period: relativedelta,
                     timestep: relativedelta,
                     spatial_res_meter: float):
    # define how to grid and which stats to calculate
    def cell_bounds(number: float):
        floor = np.floor(number/spatial_res_meter)*spatial_res_meter
        return slice(floor, floor+spatial_res_meter)
    def stats(data: pd.Series) -> pd.Series:
        median = data.median()
        mad = np.abs(data-median).median()
        return pd.Series([median, mad, data.shape[0]])
    time_axis = pd.date_range(start_datetime+pd.offsets.MonthBegin(0), end_datetime, freq=timestep)
    if time_axis.tz == None: time_axis = time_axis.tz_localize("UTC")
    # if l2_data.index[0].tz == None: l2_data.index = l2_data.index.tz_localize("UTC")
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
            parent_cell.drop(index=subset.index, inplace=True)
            print("# subset data:", subset.shape[0])
            results_list = [None]*aggregation_period.months
            for i in range(aggregation_period.months):
                results_list[i] = subset.groupby(subset.index.get_level_values("time")-pd.offsets.QuarterBegin(1, normalize=True)+pd.DateOffset(months=i)).apply(stats)
            result = pd.concat(results_list).unstack().sort_index().rename(columns={0: "med_elev_diff", 1: "mad_elev_diff", 2: "cnt_elev_diff"})#, inplace=True
            result = result.loc[time_axis.join(result.index, how="inner")]
            gridded_list.append(pd.concat([result], keys=[(x_slice.start,y_slice.start)], names=["x", "y", "time"]))
        # consider saving a backup to disk after each parent_cell
    return pd.concat(gridded_list).to_xarray()
__all__.append("med_mad_cnt_grid")
