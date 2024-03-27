import dask.dataframe
from dateutil.relativedelta import relativedelta
# import numba
import numpy as np
import os
import pandas as pd
import shapely
import rasterio.warp
import rioxarray as rioxr
import xarray as xr

from . import l2
from .misc import *

__all__ = list()
    

# numba does not do help here easily. using the numpy functions is as fast as it gets.
def med_iqr_cnt(data):
    quartiles = np.quantile(data, [.25, .5, .75])
    return pd.DataFrame([[quartiles[1], quartiles[2]-quartiles[0], len(data)]], columns=["_median", "_iqr", "_count"])
__all__.append("med_iqr_cnt")


def build_dataset(region_of_interest: str|shapely.Polygon,
                  start_datetime: str|pd.Timestamp,
                  end_datetime: str|pd.Timestamp, *,
                  l2_type: str = "swath",
                  timestep_months: int = 1,
                  window_ntimesteps: int = 3,
                  spatial_res_meter: float = 500,
                  agg_func_and_meta: tuple[callable, dict] = (med_iqr_cnt,
                                                              {"_median": "f8", "_iqr": "f8", "_count": "i8"}),
                  **l2_from_id_kwargs):
    if window_ntimesteps%2 - 1:
        old_window = window_ntimesteps
        window_ntimesteps = (window_ntimesteps//2+1)
        warnings.warn(f"The window should be a uneven number of time steps. You asked for {old_window}, but it has "+ f"been changed to {window_ntimesteps}.")
    # ! end time step should be included.
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    print("Building a gridded dataset of elevation estimates for",
          "the region "+region_of_interest if isinstance(region_of_interest, str) else "a custom area",
          f"from {start_datetime} to {end_datetime} every {timestep_months} months for",
          f"a rolling window of {window_ntimesteps} time steps.")
    if "buffer_region_by" not in locals():
        # buffer_by defaults to 30 km to not miss any tracks. Usually,
        # 10 km should do.
        buffer_region_by = 30_000
    time_buffer_months = (window_ntimesteps*timestep_months)//2
    ext_t_axis = pd.date_range(start_datetime-pd.DateOffset(months=time_buffer_months),
                               end_datetime+pd.DateOffset(months=time_buffer_months),
                               freq=f"{timestep_months}MS",
                               ).astype("int64")
    cs_tracks = load_cs_ground_tracks(region_of_interest, start_datetime, end_datetime,
                                      buffer_period_by=relativedelta(months=time_buffer_months),
                                      buffer_region_by=buffer_region_by)
    print("First and last available ground tracks are on",
          f"{cs_tracks.index[0]} and {cs_tracks.index[-1]}, respectively.,",
          f"{cs_tracks.shape[0]} tracks in total."
          "\n[note] Run update_cs_ground_tracks, optionally with `full=True` or",
          "`incremental=True`, if you local ground tracks store is not up to",
          "date. Consider pulling the latest version from the repository.")

    # ! exclude data out of regions total_bounds in l2.from_id (?possible/logically consistent?)
    print("Storing the essential L2 data in hdf5, downloading and",
          "processing L1b files if not available...")
    if isinstance(region_of_interest, str):
        region_id = region_of_interest
    else:
        region_id = "_".join([region_of_interest.centroid.x, region_of_interest.centroid.y])
    cache_path = os.path.join(data_path, "tmp", region_id)
    l2.from_id(cs_tracks.index, save_or_return="save", cache=cache_path,
               **filter_kwargs(l2.from_id, l2_from_id_kwargs, blacklist=["save_or_return", "cache"]))

    print("Gridding the data...")
    # one could drop some of the data before gridding. however, excluding
    # off-glacier data is expensive and filtering large differences to the
    # DEM can hide issues while statistics like the median and the IQR
    # should be fairly robust.
    l2_ddf = dask.dataframe.read_hdf(cache_path, l2_type, sorted_index=True)
    l2_ddf = l2_ddf.loc[ext_t_axis[0]:ext_t_axis[-1]]
    l2_ddf = l2_ddf.repartition(npartitions=3*len(os.sched_getaffinity(0)))

    l2_ddf[["x", "y"]] = (l2_ddf[["x", "y"]]//spatial_res_meter+.5)*spatial_res_meter
    l2_ddf["roll_0"] = l2_ddf.index.map_partitions(pd.cut, bins=ext_t_axis, right=False, labels=False, include_lowest=True)
    for i in range(1, window_ntimesteps):
        l2_ddf[f"roll_{i}"] = l2_ddf.map_partitions(lambda df: df.roll_0-i).persist()
    for i in range(window_ntimesteps):
        l2_ddf[f"roll_{i}"] = l2_ddf[f"roll_{i}"].map_partitions(lambda series: series.astype("i4")//window_ntimesteps)

    roll_res = [None]*window_ntimesteps
    for i in range(window_ntimesteps):
        roll_res[i] = l2_ddf.rename(columns={f"roll_{i}": "time_idx"}).groupby(["time_idx", "x", "y"], sort=False).h_diff.apply(agg_func_and_meta[0], meta=agg_func_and_meta[1]).persist()
    for i in range(window_ntimesteps):
        roll_res[i] = roll_res[i].compute().droplevel(3, axis=0)
        roll_res[i].index = roll_res[i].index.set_levels(
            (roll_res[i].index.levels[0]*window_ntimesteps+i+1), level=0).rename("time", level=0)
        
    l3_data = pd.concat(roll_res).sort_index()\
                                 .loc[(slice(0,len(ext_t_axis)-1),slice(None),slice(None)),:]
    l3_data.index = l3_data.index.remove_unused_levels()
    l3_data.index = l3_data.index.set_levels(
            ext_t_axis[l3_data.index.levels[0]].astype("datetime64[ns]"), level=0)
    l3_data = l3_data.query(f"time >= '{start_datetime}' and time <= '{end_datetime}'")
    # fill x and y such that they are continuous
    # otherwise, issues arise when working with the data. occurs because
    # data is not reduced to glacierized region (but could in theory always
    # occur anyway)
    l3_data = fill_missing_coords(l3_data.to_xarray()).rio.write_crs(3413)
    l3_data.to_netcdf(build_path(region_id, timestep_months, spatial_res_meter))
    return l3_data
__all__.append("build_dataset")


def build_path(region_of_interest, timestep_months, spatial_res_meter, aggregation_period):
    if not isinstance(region_of_interest, str):
        region_id = find_region_id(region_of_interest)
    else:
        region_id = region_of_interest
    if timestep_months != 1:
        timestep_str = str(timestep_months)+"-"
    else:
        timestep_str = ""
    timestep_str += "monthly"
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


def fill_missing_coords(l3_data):
    # inspired by user9413641
    # https://stackoverflow.com/questions/68207994/fill-in-missing-index-positions-in-xarray-dataarray
    coords = {k: range(l3_data[k].min().values, l3_data[k].max().values+1, l3_data[k].diff(k).min().values)
              for k in ["x", "y"]}
    return l3_data.reindex(coords, fill_value=np.nan)
__all__.append("fill_missing_coords")


def fill_voids(l3_data):
    # for now, this is a very specific function despite its name. it expects
    # l3_data to have the dimensions x, y, and time, that it covers an RGI
    # o2 region, and that it contains the stats: median_h_diff, IQR_h_diff,
    # and data_count.
    print("... reading reference DEM")
    with get_dem_reader() as dem_reader:
        with rioxr.open_rasterio(dem_reader) as ref_dem:
            ref_dem = ref_dem.rio.clip_box(*l3_data.rio.bounds()).squeeze()
    l3_data["ref_elev"] = ref_dem.rio.reproject_match(l3_data, resampling=rasterio.warp.Resampling.average).transpose("x", "y")
    # figure out region. limited to o2 meanwhile
    print("... loading basin outlines")
    left, lower, right, upper = l3_data.rio.transform_bounds(4326)
    o2code = find_region_id(shapely.Point(left+(right-left)/2, lower+(upper-lower)/2), scope="o2")
    basin_shapes = load_o2region(o2code, product="glaciers").to_crs(l3_data.rio.crs)
    print("... assigning basin ids to grid cells")
    l3_crs = l3_data.rio.crs
    l3_data["basin_id"] = xr.DataArray(np.zeros_like(l3_data.ref_elev, dtype="i4"),
                                       coords={k: v for k, v in l3_data.coords.items() if k in ["x", "y"]}
                                       ).rio.write_crs(l3_crs)
    for i in range(len(basin_shapes)):
        try:
            subset = l3_data.basin_id.rio.clip(basin_shapes.iloc[[i]].geometry)
        except rioxr.exceptions.NoDataInBounds:
            continue
        else:
            subset = xr.where(subset.isnull(), subset, int(basin_shapes.iloc[i].rgi_id.split("-")[-1]))
            aligned = xr.align(subset, l3_data.basin_id, join="right")#, copy=False
            l3_data["basin_id"] = xr.where(aligned[0].isnull(), l3_data.basin_id, aligned[0]).rio.write_crs(l3_crs)
    l3_ddf = l3_data.to_dask_dataframe()
    l3_ddf = l3_ddf.map_partitions(lambda df: df[df.basin_id>0])
    del l3_data
    print("... interpolating data per basin")
    global filled_basin_ids
    filled_basin_ids = []
    l3_ddf = l3_ddf.groupby(l3_ddf.basin_id).apply(interpolation_wrapper, meta=l3_ddf.partitions[0].head(0)).compute()
    print("... interpolating remaining voids")
    # the remaining voids are filled based on termination type and location
    basin_shapes["voids_filled"] = [basin_id in filled_basin_ids for basin_id in basin_shapes.rgi_id.apply(lambda x: int(x.split("-")[-1]))]
    for basin_tt_group in basin_shapes[~basin_shapes.voids_filled].groupby("term_type"):
        # cut latitude into degree slices
        n_lat_bins = max(1, round(basin_tt_group[1].cenlat.max()-basin_tt_group[1].cenlat.min()))
        for basin_lat_group in basin_tt_group[1].groupby(pd.cut(basin_tt_group[1].cenlat, bins=n_lat_bins)):
            # similarly, cut longitude
            n_lon_bins = max(1, round((basin_lat_group[1].cenlon.max()-basin_lat_group[1].cenlon.min())
                                      * np.cos(np.deg2rad(basin_lat_group[0].mid))))
            for basin_lon_group in basin_lat_group[1].groupby(pd.cut(basin_lat_group[1].cenlon, bins=n_lon_bins)):
                # use all cells with matching term_type in proximity as reference
                frame = basin_lon_group[1].unary_union.bounds
                matching_tt = basin_shapes.clip(frame).rgi_id.apply(lambda x: int(x.split("-")[-1])).to_list()
                # below, cutting to the frame is probably not really necessary because
                # matching_tt is already constrained
                subset = l3_ddf.loc[lambda df: df.basin_id.isin(matching_tt)].loc[lambda df: df.x >= frame[0]]\
                               .loc[lambda df: df.y >= frame[1]].loc[lambda df: df.x <= frame[2]]\
                               .loc[lambda df: df.y <= frame[3]]
                subset = interpolate_hypsometrically_poly3(subset)
                l3_ddf.loc[subset.index] = subset
                basin_shapes.loc[basin_lon_group[1].index,"voids_filled"] = True
    l3_data = l3_ddf.set_index(["time", "x", "y"])[["median_h_diff", "IQR_h_diff", "data_count"]].to_xarray().rio.write_crs(l3_crs)
    return l3_data
__all__.append("fill_voids")


def interpolate_hypsometrically_poly3(df):
    # helper function for `fill_voids`. as it is now, there is a number of
    # strict requirements on df: it has to have the columns median_h_diff,
    # IQR_h_diff, data_count, and ref_elev. in ref_elev, there should be no
    # no-data values where median_h_diff is valid.
    weights = df.data_count**.5/df.IQR_h_diff
    # use only grid cells based on a minimum number of elevation estimates
    # and that are not too far of
    weights.loc[~(df.data_count>3)] = 0
    weights.loc[~(np.abs(df.median_h_diff)<150)] = 0
    # abort if too little data. necessary to prevent errors but also introduces data gaps
    if sum(weights>0) <= 20:
        return df
    weights = weights/weights.loc[weights>0].mean()
    # first fit
    coeffs = np.polyfit(df.ref_elev, df.median_h_diff, 3, w=weights)
    residuals = np.polyval(coeffs, df.ref_elev) - df.median_h_diff.fillna(0)
    # find and remove outlier
    outlier_mask = flag_outliers(residuals[weights>0], deviation_factor=5)
    # print("dropping", sum(outlier_mask), f", {sum(~outlier_mask)} remain")
    weights.loc[weights>0] = ~outlier_mask.values * weights.loc[weights>0]
    # fit again
    coeffs = np.polyfit(df.ref_elev, df.median_h_diff, 3, w=weights)
    lowest = df.ref_elev[weights>0].min()
    highest = df.ref_elev[weights>0].max()
    weights = weights.reindex(df.index, fill_value=0)
    df.loc[weights==0,"median_h_diff"] = np.polyval(coeffs, df.ref_elev[weights==0])
    df.loc[df.ref_elev<lowest,"median_h_diff"] = np.polyval(coeffs, lowest)
    df.loc[df.ref_elev>highest,"median_h_diff"] = np.polyval(coeffs, highest)
    df.loc[weights==0,"IQR_h_diff"] = np.inf
    df.loc[weights==0,"data_count"] = 0
    return df
__all__.append("interpolate_hypsometrically_poly3")


def interpolation_wrapper(df):
    # helper function for `fill_voids`
    n_data_per_t = df[df.data_count>3].time.groupby(df.time).count()
    if np.mean(n_data_per_t)/8 < 5:
        return df
    df = df.groupby(df.time).apply(interpolate_hypsometrically_poly3) #.reset_index()(names=""), include_groups=False
    df.index = df.index.levels[1]
    global filled_basin_ids
    filled_basin_ids.append(df.iat[0,-1])
    return df

__all__ = sorted(__all__)
