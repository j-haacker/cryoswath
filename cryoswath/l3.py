import dask.dataframe
import dask.distributed
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import h5py
# import numba
import numpy as np
import os
import pandas as pd
from pyproj.crs import CRS
import shapely
import shutil
import sys
import rasterio.warp
import rioxarray as rioxr
import xarray as xr

from . import l2
from .misc import *
from .gis import ensure_pyproj_crs, find_planar_crs

__all__ = list()
    

def append_basin_id(ds: xr.DataArray|xr.Dataset,
                    basin_gdf: gpd.GeoDataFrame = None,
                    ) -> xr.Dataset:
    if basin_gdf is None:
        raise NotImplementedError("Automatic basin loading is not yet implemented.")
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    ds["basin_id"] = xr.DataArray(-1,
                                  coords={k: v for k, v in ds.coords.items() if k in ["x", "y"]},
                                  dims=["x", "y"],
                                  attrs={"_FillValue": -9999})
    for i in range(len(basin_gdf)):
        try:
            subset = ds.basin_id.rio.clip(basin_gdf.iloc[[i]].geometry)
        except rioxr.exceptions.NoDataInBounds:
            continue
        subset = xr.where(subset==subset._FillValue, ds.basin_id.loc[dict(x=subset.x, y=subset.y)], int(basin_gdf.iloc[i].rgi_id.split("-")[-1])).rio.write_crs(ds.rio.crs)
        ds["basin_id"].loc[dict(x=subset.x, y=subset.y)] = subset
    ds["basin_id"] = xr.where(ds.basin_id==-1, ds.basin_id._FillValue, ds.basin_id)
    return ds
__all__.append("append_basin_id")
    

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
                  crs: CRS|int = None,
                  reprocess: bool = False,
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
        region_of_interest = load_glacier_outlines(region_id)
    else:
        region_id = "_".join([region_of_interest.centroid.x, region_of_interest.centroid.y])
    cache_path = os.path.join(data_path, "tmp", region_id)
    if crs is None:
        crs = find_planar_crs(region_id=region_id)
    else:
        crs = ensure_pyproj_crs(crs)
    bbox = gpd.GeoSeries(
        shapely.box(*gpd.GeoSeries(region_of_interest, crs=4326).to_crs(crs).bounds.values[0]),
        crs=crs)
    # l2 backs up the cache when writing to it. however, there should not be a backup, yet. if there is, throw an error
    if os.path.isfile(cache_path+"__backup"):
        raise Exception(f"Backup exists unexpectedly at {cache_path+'__backup'}. This may point to a running process. If this is a relict, remove it manually.")
    try:
        l2.from_id(cs_tracks.index, save_or_return="save", cache=cache_path, crs=crs, bbox=bbox,
                   **filter_kwargs(l2.from_id, l2_from_id_kwargs, blacklist=["save_or_return", "cache"]))
    finally:
        # remove l2's cache backup. it is not needed as no more writing takes
        # place but it occupies some 10 Gb disk space.
        if os.path.isfile(cache_path+"__backup"):
            os.remove(cache_path+"__backup")
    outfilepath = build_path(region_id, timestep_months, spatial_res_meter)
    if not reprocess and os.path.isfile(outfilepath):
        previously_processed_l3 = xr.open_dataset(outfilepath, decode_coords="all")
        print(previously_processed_l3)
        print(previously_processed_l3.rio.crs) # no crs? that shouldn't be the case! 
    node_list = []
    def guide_hdf_node(name, node):
        nonlocal node_list
        if not isinstance(node, h5py.Dataset) and "_i_table" in node:
            if "previously_processed_l3" in locals():
                x_range, y_range = node.name.split("/")[-2:]
                x0, x1 = [int(item) for item in x_range.split("_")[-2:]]
                y0, y1 = [int(item) for item in y_range.split("_")[-2:]]
                # print(x0, x1, y0, y1)
                if not previously_processed_l3._median.sel(x=slice(x0, x1), y=slice(y0, y1)).isnull().all():
                    print("cell", node.name, "will be skipped. data is present")
                    return None
            node_list.append(node.name)
    with h5py.File(cache_path, "r") as h5:
        h5["swath"].visititems(guide_hdf_node)
    print("processing queue contains:\n", "\n".join(node_list))
    print("\nGridding the data. Each chunk at a time...")
    for node_name in node_list:
        print("-----\n\nnext chunk:", node_name)
        # reading from hdf has issues acquiring the lock. Since no writing takes
        # place, it should be safe to turn this off. This can potentially be
        # resumed in future.
        l2_ddf = dask.dataframe.read_hdf(cache_path, node_name, sorted_index=True, lock=False, mode="r", )
        l2_ddf = l2_ddf.loc[ext_t_axis[0]:ext_t_axis[-1]]
        # one could drop some of the data before gridding. however, excluding
        # off-glacier data is expensive and filtering large differences to the
        # DEM can hide issues while statistics like the median and the IQR
        # should be fairly robust.
        if len(l2_ddf.index) != 0:
            l2_ddf = l2_ddf.repartition(npartitions=3*len(os.sched_getaffinity(0)))
            l2_ddf[["x", "y"]] = ((l2_ddf[["x", "y"]]//spatial_res_meter+.5)*spatial_res_meter).astype("i4")
            l2_ddf["roll_0"] = l2_ddf.index.map_partitions(pd.cut, bins=ext_t_axis, right=False, labels=False, include_lowest=True)
            # note on the for-loops:
            #     because of the late-binding python behavior, one or the other way the
            #     counting index must be defined at place (as opposed to when dask tries
            #     to calculate the values because then all the indeces have the same
            #     value). the chosen way is defining a function which creates a new
            #     namespace (in which the index is copied and will not be changed from
            #     outside).
            for i in range(1, window_ntimesteps):
                def local_closure(roll_iteration):
                    return l2_ddf.map_partitions(lambda df: df.roll_0-roll_iteration)
                l2_ddf[f"roll_{i}"] = local_closure(i)
            for i in range(window_ntimesteps):
                def local_closure(roll_iteration):
                    return l2_ddf[f"roll_{i}"].map_partitions(lambda series: series.astype("i4")//window_ntimesteps)
                l2_ddf[f"roll_{i}"] = local_closure(i)
            # results_list actually is a graph_list until .compute()
            results_list = [None]*window_ntimesteps
            for i in range(window_ntimesteps):
                def local_closure(roll_iteration):
                    return l2_ddf.rename(columns={f"roll_{i}": "time_idx"}).groupby(["time_idx", "x", "y"], sort=False).h_diff.apply(agg_func_and_meta[0], meta=agg_func_and_meta[1])
                results_list[i] = local_closure(i)
            for i in range(window_ntimesteps):
                results_list[i] = results_list[i].compute()
                results_list[i] = results_list[i].droplevel(3, axis=0)
                results_list[i].index = results_list[i].index.set_levels(
                    (results_list[i].index.levels[0]*window_ntimesteps+i+1), level=0).rename("time", level=0)
            l3_data = pd.concat(results_list).sort_index().loc[(slice(0,len(ext_t_axis)-1),slice(None),slice(None)),:]
            l3_data.index = l3_data.index.remove_unused_levels()
            l3_data.index = l3_data.index.set_levels(
                    ext_t_axis[l3_data.index.levels[0]].astype("datetime64[ns]"), level=0)
            l3_data = l3_data.sort_index()
            l3_data = l3_data.query(f"time >= '{start_datetime}' and time <= '{end_datetime}'")
            # # to_xarray threw errors in the past because of duplicate indices. there
            # # should never(?) be any. activate the below to debug should this
            # # problem occur again.
            # duplicates = l3_data.index.duplicated(keep=False)
            # print(l3_data.index[duplicates])
            # print(l3_data.tail(30))
            
            # fill x and y such that they are continuous
            # otherwise, issues arise when working with the data. occurs because
            # data is not reduced to glacierized region (but could in theory always
            # occur anyway)
            try:
                l3_data = fill_missing_coords(l3_data.to_xarray())
            except:
                print(l3_data)
                print(l3_data.to_xarray())
                raise
            # merge with previous data if there are any
            if "previously_processed_l3" in locals():
                # unfortunately, combining datasets can be nasty:
                # `xarray.combine_by_coords()` uses the coordinates in order, leading to
                # issues if, e.g., x starts earlier for one, but y starts earlier for
                # the other dataset. the below attempts different orders; however,
                # success is not guaranteed :/
                try:
                    previously_processed_l3 = xr.combine_nested([previously_processed_l3, l3_data], concat_dim=None)
                except ValueError:
                    previously_processed_l3 = xr.combine_nested([l3_data, previously_processed_l3], concat_dim=None)
            else:
                previously_processed_l3 = l3_data
            # save/backup result
            tmp_path = os.path.join(data_path, "tmp", f"{region_id}_l3")
            # try to write new data to file. if anything goes wrong, restore. is this sufficiently safe?
            try:
                if os.path.isfile(outfilepath):
                    shutil.move(outfilepath, tmp_path)
                previously_processed_l3.rio.write_crs(crs).to_netcdf(outfilepath)
                print(f"processed and stored cell", node_name)
            except:
                shutil.move(tmp_path, outfilepath)
            else:
                os.remove(tmp_path)
    return previously_processed_l3
__all__.append("build_dataset")


def build_path(region_of_interest, timestep_months, spatial_res_meter, aggregation_period = None):
    # ! implement parsing aggregation period
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
              for k in ["x", "y"] if len(l3_data[k])>2}
    return l3_data.reindex(coords, fill_value=np.nan)
__all__.append("fill_missing_coords")


def fill_voids(l3_data):
    # for now, this is a very specific function despite its name. it expects
    # l3_data to have the dimensions x, y, and time, that it covers an RGI
    # o2 region, and that it contains the stats: _median, _iqr,
    # and _count.
    # figure out region. limited to o2 meanwhile
    print("... loading basin outlines")
    left, lower, right, upper = l3_data.rio.transform_bounds(4326)
    o2code = find_region_id(shapely.Point(left+(right-left)/2, lower+(upper-lower)/2), scope="o2")
    # print(o2code)
    basin_shapes = load_o2region(o2code, product="glaciers").to_crs(l3_data.rio.crs)
    print("... reading reference DEM")
    # finding a latitude to determine the reference DEM like below may be prone to bugs
    with get_dem_reader(l3_data._median[0].transpose("y", "x").rio.reproject(4326).y.values[0]) as dem_reader:
        with rioxr.open_rasterio(dem_reader) as ref_dem:
            ref_dem = ref_dem.rio.clip_box(*l3_data._median[0].rio.reproject(ref_dem.rio.crs).rio.bounds()).squeeze()
    # print(l3_data.dims, ref_dem.dims)
    l3_data["ref_elev"] = ref_dem.rio.reproject_match(l3_data, resampling=rasterio.warp.Resampling.average).transpose("x", "y").rio.clip(basin_shapes.geometry)
    l3_data.ref_elev.attrs.update({"_FillValue": np.nan})
    l3_data["ref_elev"] = xr.where(l3_data.ref_elev < 100, np.nan, l3_data.ref_elev)
    # from matplotlib import pyplot as plt
    # l3_data.ref_elev.T.plot(robust=True)
    # plt.show()
    print("num nan in ref_elev", l3_data.ref_elev.isnull().sum())
    print("num other in ref_elev", np.logical_or(l3_data.ref_elev<-100, l3_data.ref_elev>3000).sum())
    # print(l3_data.ref_elev)
    # l3_data.ref_elev.rio.clip(basin_shapes.geometry).T.plot(vmin=-10, vmax=1300)
    # basin_shapes.boundary.plot(ax=plt.gca())
    # plt.show()
    # ref_dem.rio.reproject_match(l3_data, resampling=rasterio.warp.Resampling.average).rio.clip(basin_shapes.geometry).plot(vmin=-10, vmax=1300)
    # basin_shapes.boundary.plot(ax=plt.gca())
    # plt.show()
    # ref_dem.rio.clip(basin_shapes.to_crs(ref_dem.rio.crs).geometry).plot(vmin=-10, vmax=1300)
    # basin_shapes.to_crs(ref_dem.rio.crs).boundary.plot(ax=plt.gca())
    # plt.show()
    print("... assigning basin ids to grid cells")
    l3_crs = l3_data.rio.crs
    # print(l3_crs)
    l3_data["basin_id"] = xr.DataArray(np.zeros_like(l3_data.ref_elev, dtype="i4"),
                                       coords={k: v for k, v in l3_data.coords.items() if k in ["x", "y"]}
                                       ).rio.write_crs(l3_crs)
    # print(l3_data.basin_id)
    for i in range(len(basin_shapes)):
        try:
            subset = l3_data.basin_id.rio.clip(basin_shapes.iloc[[i]].geometry)
        except rioxr.exceptions.NoDataInBounds:
            continue
        else:
            subset = xr.where(subset.isnull(), subset, int(basin_shapes.iloc[i].rgi_id.split("-")[-1]))
            aligned = xr.align(subset, l3_data.basin_id, join="right")#, copy=False
            l3_data["basin_id"] = xr.where(aligned[0].isnull(), l3_data.basin_id, aligned[0]).rio.write_crs(l3_crs)
    l3_df = l3_data.to_dataframe()
    l3_df = l3_df[l3_df.basin_id>0]
    print("... interpolating data per basin")
    l3_df = l3_df.groupby(l3_df.basin_id).apply(interpolation_wrapper)
    print("l3_ddf shape after basin void filling:\n", l3_df.head(0))
    l3_df = l3_df.droplevel(0)
    print("l3_ddf shape after dropping lev0:\n", l3_df.head(0))
    print("... interpolating remaining voids")
    # the remaining voids are filled based on termination type and location
    for basin_tt_group in basin_shapes.groupby("term_type"):
        # cut latitude into degree slices
        n_lat_bins = max(1, round(basin_tt_group[1].cenlat.max()-basin_tt_group[1].cenlat.min()))
        # below, `observed=True` to grant compatibility with future pandas versions.
        for basin_lat_group in basin_tt_group[1].groupby(pd.cut(basin_tt_group[1].cenlat, bins=n_lat_bins),
                                                         observed=True):
            # similarly, cut longitude
            n_lon_bins = max(1, round((basin_lat_group[1].cenlon.max()-basin_lat_group[1].cenlon.min())
                                      * np.cos(np.deg2rad(basin_lat_group[0].mid))))
            for basin_lon_group in basin_lat_group[1].groupby(pd.cut(basin_lat_group[1].cenlon, bins=n_lon_bins),
                                                              observed=True):
                # use all cells with matching term_type in proximity as reference
                frame = basin_lon_group[1].unary_union.bounds
                matching_tt = basin_shapes.clip(frame).rgi_id.apply(lambda x: int(x.split("-")[-1])).to_list()
                subset = l3_df.loc[l3_df.basin_id.isin(matching_tt)]
                subset = interpolation_wrapper(subset)
                l3_df.loc[subset.index] = subset
    l3_data = xr.merge([l3_df[["_median", "_iqr", "_count"]].to_xarray(), l3_data], join="right", compat="override")
    return l3_data
__all__.append("fill_voids")


def interpolate_hypsometrically_poly3(df):
    # helper function for `fill_voids`. as it is now, there is a number of
    # strict requirements on df: it has to have the columns _median,
    # _iqr, _count, and ref_elev. in ref_elev, there should be no
    # no-data values where _median is valid.
    df_valid_ref_elev = df[~df.ref_elev.isnull()]
    if df_valid_ref_elev.empty:
        return df
    df_only_valid = df_valid_ref_elev[~df_valid_ref_elev.isnull().any(axis=1)]
    if df_only_valid.empty:
        return df
    weights = df_only_valid._count**.5/df_only_valid._iqr
    # use only grid cells based on a minimum number of elevation estimates
    # and that are not too far of
    weights.loc[~(df_only_valid._count>3)] = 0
    weights.loc[~(np.abs(df_only_valid._median)<150)] = 0
    # abort if too little data. necessary to prevent errors but also introduces data gaps
    if sum(weights>0) <= 20:
        return df
    # also, abort if there isn't anything to do
    if not any(weights==0):
        return df
    weights = weights/weights.loc[weights>0].mean()
    # first fit
    x0 = df_only_valid.ref_elev.mean()
    coeffs = np.polyfit(df_only_valid.ref_elev-x0, df_only_valid._median, 3, w=weights)
    residuals = np.polyval(coeffs, df_only_valid.ref_elev-x0) - df_only_valid._median
    # find and remove outlier
    outlier_mask = flag_outliers(residuals[weights>0], deviation_factor=5)

    # # debugging tool
    # import matplotlib.pyplot as plt
    # # print(df_only_valid.ref_elev[weights>0], df_only_valid._median[weights>0], 1/weights[weights>0], '_')
    # plt.errorbar(df_only_valid.ref_elev[weights>0], df_only_valid._median[weights>0], yerr=.1/weights[weights>0], fmt='_')
    # plt.plot(df_only_valid.ref_elev[weights>0][outlier_mask], df_only_valid._median[weights>0][outlier_mask], 'rx')
    # pl_x = np.arange(0, 2001, 100)
    # plt.plot(pl_x, np.polyval(coeffs, pl_x), '-')
    # plt.show()

    weights.loc[weights>0] = ~outlier_mask.values * weights.loc[weights>0]
    # fit again
    coeffs = np.polyfit(df_only_valid.ref_elev-x0, df_only_valid._median, 3, w=weights)
    lowest = df_only_valid.ref_elev[weights>0].min()
    highest = df_only_valid.ref_elev[weights>0].max()
    # from here, weights are only used as mask. to-be-filled: 0, missing ref_elev: -1
    weights = weights.reindex(df_valid_ref_elev.index, fill_value=0)
    weights = weights.reindex(df.index, fill_value=-1)
    df.loc[weights==0,"_median"] = np.polyval(coeffs, df.ref_elev[weights==0]-x0)
    df.loc[df.ref_elev<lowest,"_median"] = np.polyval(coeffs, lowest-x0)
    df.loc[df.ref_elev>highest,"_median"] = np.polyval(coeffs, highest-x0)
    df.loc[weights==0,"_iqr"] = np.inf
    df.loc[weights==0,"_count"] = 0
    return df
__all__.append("interpolate_hypsometrically_poly3")


def interpolation_wrapper(df):
    # helper function for `fill_voids`
    n_data_per_t = df[df._count>3].groupby(level=[0]).size()
    if not np.mean(n_data_per_t)/8 >= 5:
        return df
    df = df.groupby(level=[0]).apply(interpolate_hypsometrically_poly3)
    df = df.droplevel(0)
    return df

__all__ = sorted(__all__)
