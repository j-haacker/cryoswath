from dateutil.relativedelta import relativedelta
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import shapely

from . import l2
from .misc import *

__all__ = list()


def build_dataset(region_of_interest: str|shapely.Polygon,
                  start_datetime: str|pd.Timestamp,
                  end_datetime: str|pd.Timestamp, *,
                  aggregation_period: relativedelta = relativedelta(months=3),
                  timestep: relativedelta = relativedelta(months=1),
                  spatial_res_meter: float = 500,
                  **kwargs):
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    print("Building a gridded dataset of elevation estimates for the region",
          f"{region_of_interest} from {start_datetime} to {end_datetime} for",
          f"a rolling window of {aggregation_period} every {timestep}.")
    # if len(aggregation_period.kwds.keys()) != 1 \
    # or len(timestep.kwds.keys()) != 1 \
    # or list(aggregation_period.kwds.keys())[0] not in ["years", "months", "days"] \
    # or list(timestep.kwds.keys())[0] not in ["years", "months", "days"]:
    #     raise Exception("Only use one of years, months, days for agg_time and timestep.")
    if "buffer_region_by" not in locals():
        # buffer_by defaults to 30 km to not miss any tracks. Usually,
        # 10 km should do.
        buffer_region_by = 30_000
    time_buffer = (aggregation_period-timestep)/2
    cs_tracks = load_cs_ground_tracks(region_of_interest, start_datetime, end_datetime,
                                      buffer_period_by=time_buffer,buffer_region_by=buffer_region_by)
    print("First and last available ground tracks are on",
          f"{cs_tracks.index[0]} and {cs_tracks.index[-1]}, respectively.,",
          f"{cs_tracks.shape[0]} tracks in total.")
    print("Run update_cs_ground_tracks, optionally with `full=True` or",
          "`incremental=True`, if you local ground tracks store is not up to",
          "date. Consider pulling the latest version from the repository.")
    # I believe passing loading l2 data to the function prevents copying
    # on .drop. an alternative would be to define l2_data nonlocal
    # within the gridding function
    l3_data =  med_mad_cnt_grid(l2.from_id(cs_tracks.index, **filter_kwargs(l2.from_id, kwargs)),
                                start_datetime=start_datetime, end_datetime=end_datetime,
                                aggregation_period=aggregation_period, timestep=timestep,
                                spatial_res_meter=spatial_res_meter)
    l3_data.to_netcdf(build_path(region_of_interest, timestep, spatial_res_meter, aggregation_period))
    return l3_data
__all__.append("build_dataset")


def build_path(region_of_interest, timestep, spatial_res_meter, aggregation_period):
    region_id = find_region_id(region_of_interest)
    if list(timestep.kwds.values())[0]!=1:
        timestep_str = str(list(timestep.kwds.values())[0])+"-"
    else:
        timestep_str = ""
    timestep_str += list(timestep.kwds.keys())[0][:-1]+"ly"
    if spatial_res_meter == 1000:
        spatial_res_str = "1km"
    elif np.floor(spatial_res_meter/1000) < 2:
        spatial_res_str = f"{spatial_res_meter}m"
    else:
        # if the trailing ".0" should be omitted, that needs to be implemented here
        spatial_res_str = f"{round(spatial_res_meter/1000, 1)}km"
    return os.path.join(data_path, "L3", "_".join(
        [region_id, timestep_str, spatial_res_str+".nc"]))
__all__.append("build_path")
    

def med_mad_cnt_grid(l2_data: gpd.GeoDataFrame, *,
                     start_datetime: pd.Timestamp,
                     end_datetime: pd.Timestamp,
                     aggregation_period: relativedelta,
                     timestep: relativedelta,
                     spatial_res_meter: float):
    def stats(data: pd.Series) -> pd.Series:
        median = data.median()
        mad = np.abs(data-median).median()
        return pd.Series([median, mad, data.shape[0]])
    time_axis = pd.date_range(start_datetime+pd.offsets.MonthBegin(0), end_datetime, freq=timestep)
    if time_axis.tz == None: time_axis = time_axis.tz_localize("UTC")
    # if l2_data.index[0].tz == None: l2_data.index = l2_data.index.tz_localize("UTC")
    def rolling_stats(data):
        results_list = [None]*aggregation_period.months
        for i in range(aggregation_period.months):
            results_list[i] = data.groupby(subset.index.get_level_values("time")-pd.offsets.QuarterBegin(1, normalize=True)+pd.DateOffset(months=i)).apply(stats)
        result = pd.concat(results_list).unstack().sort_index().rename(columns={0: "med_elev_diff", 1: "mad_elev_diff", 2: "cnt_elev_diff"})#, inplace=True
        return result.loc[time_axis.join(result.index, how="inner")]
    return l2.grid(l2_data, spatial_res_meter, rolling_stats).to_xarray()
__all__.append("med_mad_cnt_grid")


__all__ = sorted(__all__)
