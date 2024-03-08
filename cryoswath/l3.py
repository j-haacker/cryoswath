import dask.dataframe
from dask.distributed import LocalCluster, Client
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import numba
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
                  l2_type: str = "swath",
                  aggregation_period: relativedelta = relativedelta(months=3),
                  timestep: relativedelta = relativedelta(months=1),
                  spatial_res_meter: float = 500,
                  **kwargs):
    # tracemalloc.start()
    # ! end time step should be included.
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    print("Building a gridded dataset of elevation estimates for",
          "the region "+region_of_interest if isinstance(region_of_interest, str) else "a custom area",
          f"from {start_datetime} to {end_datetime} for",
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
    # ! will fail for basically any custom input
    # t_axis = pd.date_range(start_datetime, end_datetime, freq=f"{timestep.months}MS")
    ext_t_axis = pd.date_range(start_datetime-pd.DateOffset(months=time_buffer.months),
                               end_datetime+pd.DateOffset(months=time_buffer.months),
                               freq=f"{timestep.months}MS",
                               )
    cs_tracks = load_cs_ground_tracks(region_of_interest, start_datetime, end_datetime,
                                      buffer_period_by=time_buffer, buffer_region_by=buffer_region_by)
    print("First and last available ground tracks are on",
          f"{cs_tracks.index[0]} and {cs_tracks.index[-1]}, respectively.,",
          f"{cs_tracks.shape[0]} tracks in total."
          "\n[note] Run update_cs_ground_tracks, optionally with `full=True` or",
          "`incremental=True`, if you local ground tracks store is not up to",
          "date. Consider pulling the latest version from the repository.")

    
    print("Storing the essential L2 data in hdf5, downloading and",
          "processing L1b files if not available...")
    if isinstance(region_of_interest, str):
        region_id = region_of_interest
    else:
        region_id = "_".join([region_of_interest.centroid.x, region_of_interest.centroid.y])
    cache_path = os.path.join(data_path, "tmp", region_id)
    l2.from_id(cs_tracks.index, save_or_return="save", cache=cache_path,
               **filter_kwargs(l2.from_id, kwargs, blacklist=["save_or_return", "cache"]))

    print("Gridding the data...")
    cluster = LocalCluster()
    client = Client(cluster)
    print(client)
    l2_ddf = dask.dataframe.read_hdf(cache_path, l2_type, sorted_index=True) # chunksize=1024
    l2_ddf = l2_ddf.set_index(l2_ddf.index.astype("datetime64[ns]"), sorted=True, sort=False)
    l2_ddf = l2_ddf.loc[ext_t_axis[0]:ext_t_axis[-1]]
    l2_ddf = l2_ddf.repartition(npartitions=3*len(os.sched_getaffinity(0)))
    print("debug0")
    # l2_ddf.visualize("dask_debug0.svg")

    l2_ddf[["x", "y"]] = l2_ddf[["x", "y"]]//spatial_res_meter*spatial_res_meter

    def resample(data: pd.DataFrame) -> pd.DataFrame:
        # print(data)
        result_list = [None]*(aggregation_period.months//timestep.months)
        for i in range(0, aggregation_period.months, timestep.months):
            # result_list[i] = data.resample(f"{aggregation_period.months}MS",
            #                             origin=pd.to_datetime("2010-07-01")-aggregation_period+pd.DateOffset(months=i)).agg(lambda series: pd.DataFrame([[*stats(series.to_numpy())]], index=[series.name]))#, raw=True, engine="numba"
            result_list[i] = data.resample(
                f"{aggregation_period.months}MS",
                origin=pd.to_datetime("2010-07-01")-aggregation_period+pd.DateOffset(months=i)
                ).agg({"median_h_diff": "median",
                       "IQR_h_diff": lambda x: x.quantile([.25, .75]).diff().iloc[1],
                       "data_count": "count"})
        # print(pd.concat(result_list, axis=0))
        return pd.concat(result_list, axis=0)
    # l3_data = l2_ddf.groupby(["x", "y"])["h_diff"].apply(resample, meta={0: "float", 1: "float", 2: "int"})#, meta={.25: "float", .5: "float", .75: "float", "count": "int"}.compute()
    l3_data = l2_ddf.groupby(["x", "y"])["h_diff"].apply(resample, meta={"median_h_diff": "float", "IQR_h_diff": "float", "data_count": "int"})#
    print("debug4")
    # l3_data.visualize("dask_debug4.svg")
    l3_data = l3_data.compute()
    print("debug6")
    # setting the time labels to begin central month. should you prefer the
    # 15th, add +pd.offsets.SemiMonthBegin() or day=15 to the DateOffset
    l3_data.index = l3_data.index.set_levels(l3_data.index.levels[2]+pd.DateOffset(months=aggregation_period.months//2), level="time")#
    l3_data = l3_data.query(f"time >= '{start_datetime}' and time <= '{end_datetime}'")
    print("debug7", l3_data)
    l3_data.to_pickle(build_path(region_id, timestep, spatial_res_meter, aggregation_period)[:-3]+".pkl")
    print("debug8: pickled dataframe")
    l3_data.to_xarray().to_netcdf(build_path(region_id, timestep, spatial_res_meter, aggregation_period))
    return l3_data
__all__.append("build_dataset")

@numba.njit(nogil=True)
def stats(data) -> tuple[float, float, int]:
    count = len(data)
    if count == 0:
        return np.nan, np.nan, count
    quartiles = np.quantile(data, [.25, .5, .75])
    return quartiles[1], quartiles[2]-quartiles[0], count


def build_path(region_of_interest, timestep, spatial_res_meter, aggregation_period):
    if not isinstance(region_of_interest, str):
        region_id = find_region_id(region_of_interest)
    else:
        region_id = region_of_interest
    # if list(timestep.kwds.values())[0]!=1:
    #     timestep_str = str(list(timestep.kwds.values())[0])+"-"
    # else:
    #     timestep_str = ""
    # timestep_str += list(timestep.kwds.keys())[0][:-1]+"ly"
    timestep_str = "monthly"
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
