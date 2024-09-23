import datetime
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import h5py
import numpy as np
import os
import pandas as pd
from pyproj.crs import CRS
import shapely
import shutil
import tqdm
import rasterio.warp
import rioxarray as rioxr
import xarray as xr

from . import l2
from .misc import *
from .gis import buffer_4326_shp, ensure_pyproj_crs, find_planar_crs

__all__ = list()
    

def append_basin_id(ds: xr.DataArray|xr.Dataset,
                    basin_gdf: gpd.GeoDataFrame = None,
                    ) -> xr.Dataset:
    if basin_gdf is None:
        raise NotImplementedError("Automatic basin loading is not yet implemented.")
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    ds["basin_id"] = xr.DataArray(-1.0, # should be float. is converted later anyway and if defined here, _FillValue
                                        # can be nan
                                  coords={k: v for k, v in ds.coords.items() if k in ["x", "y"]},
                                  dims=["x", "y"],
                                  attrs={"_FillValue": np.nan})
    for i in range(len(basin_gdf)):
        try:
            subset = ds.basin_id.rio.clip(basin_gdf.iloc[[i]].make_valid())
        except rioxr.exceptions.NoDataInBounds:
            continue
        subset = xr.where(subset.isnull(), ds.basin_id.loc[dict(x=subset.x, y=subset.y)], float(basin_gdf.iloc[i].rgi_id.split("-")[-1]))
        ds["basin_id"].loc[dict(x=subset.x, y=subset.y)] = subset
    ds["basin_id"] = xr.where(ds.basin_id==-1, ds.basin_id._FillValue, ds.basin_id)
    return ds
__all__.append("append_basin_id")
    

def append_basin_group(ds: xr.DataArray|xr.Dataset,
                       basin_gdf: gpd.GeoDataFrame = None,
                       ) -> xr.Dataset:
    if basin_gdf is None:
        raise NotImplementedError("Automatic basin loading is not yet implemented.")
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    ds["group_id"] = xr.DataArray(-1.0, # should be float. is converted later anyway and if defined here, _FillValue
                                        # can be nan
                                  coords={k: v for k, v in ds.coords.items() if k in ["x", "y"]},
                                  dims=["x", "y"],
                                  attrs={"_FillValue": np.nan})
    for basin_tt_group in basin_gdf.groupby("term_type"):
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
                try:
                    mask = ds.group_id.rio.clip(basin_lon_group[1].make_valid())
                except rioxr.exceptions.NoDataInBounds:
                    # if there is no data at all, continue
                    continue
                # construct id: sign indicates hemisphere, first digit is termination
                # type (see RGI doc; 0: land, 1: tidewater, 2: lake, 3: shelf, 9: n/a),
                # digits 2+3 are latitude, digits 4-6 are longitude east of 0 (0-360)
                term_type = basin_tt_group[0]
                lat = basin_lat_group[0].mid
                lon = basin_lon_group[0].mid
                group_id = int(f"{np.sign(lat)*term_type:.0f}{np.abs(lat):02.0f}{lon%360:03.0f}")
                mask = xr.where(mask==mask.isnull(), ds.group_id.loc[dict(x=mask.x, y=mask.y)], group_id)
                ds["group_id"].loc[dict(x=mask.x, y=mask.y)] = mask
    ds["group_id"] = xr.where(ds.group_id==-1, ds.group_id._FillValue, ds.group_id)
    return ds
__all__.append("append_basin_group")


def append_elevation_reference(geospatial_ds: xr.Dataset|xr.DataArray,
                               ref_elev_name: str = "ref_elev",
                               ) -> xr.Dataset:
    if isinstance(geospatial_ds, xr.DataArray):
        geospatial_ds = geospatial_ds.to_dataset()
    # finding a latitude to determine the reference DEM like below may be prone to bugs
    with get_dem_reader(geospatial_ds) as dem_reader:
        with rioxr.open_rasterio(dem_reader) as ref_dem:
            ref_dem = ref_dem.rio.clip_box(*geospatial_ds.rio.transform_bounds(ref_dem.rio.crs)).squeeze()
            ref_dem = xr.where(ref_dem==ref_dem._FillValue, np.nan, ref_dem).rio.write_crs(ref_dem.rio.crs)
            ref_dem.attrs.update({"_FillValue": np.nan})
    geospatial_ds[ref_elev_name] = xr.align(
        ref_dem.rio.reproject_match(geospatial_ds, resampling=rasterio.warp.Resampling.average,
                                    nodata=ref_dem._FillValue),
        geospatial_ds, join="right")[0]
    geospatial_ds[ref_elev_name].attrs.update({"_FillValue": np.nan})
    return geospatial_ds
__all__.append("append_elevation_reference")


# numba does not do help here easily. using the numpy functions is as fast as it gets.
def med_iqr_cnt(data):
    quartiles = np.quantile(data, [.25, .5, .75])
    return pd.DataFrame([[quartiles[1], quartiles[2]-quartiles[0], len(data)]], columns=["_median", "_iqr", "_count"])
__all__.append("med_iqr_cnt")


def build_dataset(region_of_interest: str|shapely.Polygon,
                  start_datetime: str|pd.Timestamp,
                  end_datetime: str|pd.Timestamp, *,
                  l2_type: str = "swath",
                  max_elev_diff: float = 150,
                  timestep_months: int = 1,
                  window_ntimesteps: int = 3,
                  spatial_res_meter: float = 500,
                  agg_func_and_meta: tuple[callable, dict] = (med_iqr_cnt,
                                                              {"_median": "f8", "_iqr": "f8", "_count": "i8"}),
                  cache_filename: str = None,
                  cache_filename_extra: str = None,
                  crs: CRS|int = None,
                  reprocess: bool = False,
                  **l2_from_id_kwargs):
    # include in docstring: function footprint = 2x resulting ds + 2Gb (min. 5Gb)
    if window_ntimesteps%2 - 1:
        old_window = window_ntimesteps
        window_ntimesteps = (window_ntimesteps//2+1)
        warnings.warn(f"The window should be a uneven number of time steps. You asked for {old_window}, but it has "+ f"been changed to {window_ntimesteps}.")
    # ! end time step should be included.
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    # this function only makes sense for multiple months, so assume input
    # was on the month scale and set end_datetime to end of month
    end_datetime = end_datetime.normalize() + pd.offsets.MonthBegin() - pd.Timedelta(1, "s")
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
        region_of_interest = load_glacier_outlines(region_id, "glaciers")
    else:
        region_id = "_".join([f"{region_of_interest.centroid.x:.0f}", f"{region_of_interest.centroid.y:.0f}"])
    if cache_filename is None:
        cache_filename = region_id
    if cache_filename_extra is not None:
        cache_filename += "_"+cache_filename_extra
    cache_fullname = os.path.join(tmp_path, cache_filename)
    if crs is None:
        crs = find_planar_crs(shp=region_of_interest)
    else:
        crs = ensure_pyproj_crs(crs)
    # cutting to actual glacier outlines takes very long. if needed, implement multiprocessing.
    # bbox = gpd.GeoSeries(
    #     shapely.box(*gpd.GeoSeries(region_of_interest, crs=4326).to_crs(crs).bounds.values[0]),
    #     crs=crs)
    # below tries to balance a large cache file with speed. it is not meant
    # to retain data in the suroundings - this is merely needed for the
    # implicit `simplify`` which would come at the cost of data if not
    # buffered
    bbox = gpd.GeoSeries(buffer_4326_shp(region_of_interest, 3_000), crs=4326).to_crs(crs)

    # l2 backs up the cache when writing to it. however, there should not be a backup, yet. if there is, throw an error
    if os.path.isfile(cache_fullname+"__backup"):
        raise Exception(f"Backup exists unexpectedly at {cache_fullname+'__backup'}. This may point to a running process. If this is a relict, remove it manually.")
    try:
        l2.from_id(cs_tracks.index, reprocess=reprocess, save_or_return="save", cache_fullname=cache_fullname, crs=crs,
                   bbox=bbox, max_elev_diff=max_elev_diff,
                   **filter_kwargs(l2.from_id, l2_from_id_kwargs,
                                   blacklist=["cache", "max_elev_diff", "save_or_return", "reprocess"]))
    finally:
        # remove l2's cache backup. it is not needed as no more writing takes
        # place but it occupies some 10 Gb disk space.
        if os.path.isfile(cache_fullname+"__backup"):
            os.remove(cache_fullname+"__backup")
    outfilepath = build_path(region_id, timestep_months, spatial_res_meter)
    if not reprocess and os.path.isfile(outfilepath):
        with xr.open_dataset(outfilepath, decode_coords="all") as ds:
            previously_processed_l3 = ds.load().copy()
    region_of_interest = gpd.GeoSeries(region_of_interest, crs=4326).to_crs(crs).make_valid().simplify(1_000).iloc[0]
    node_list = []
    def collect_chunk_names(name, node):
        nonlocal node_list
        name_parts = name.split("/")
        if not isinstance(node, h5py.Group) or len(name_parts) < 2 or not name_parts[-2].startswith("x_"):
            return
        chunk_name = name_parts[:2]
        if chunk_name not in node_list:
            x_range, y_range = chunk_name
            x0, x1 = [int(item) for item in x_range.split("_")[-2:]]
            y0, y1 = [int(item) for item in y_range.split("_")[-2:]]
            if x0 > region_of_interest.bounds[2] \
            or x1 < region_of_interest.bounds[0] \
            or y0 > region_of_interest.bounds[3] \
            or y1 < region_of_interest.bounds[1]:
                return
            if not shapely.box(x0, y0, x1, y1).intersects(region_of_interest):
                print("cell", chunk_name, "will be skipped. cell does not intersect current region")
            elif "previously_processed_l3" in locals() \
                    and not previously_processed_l3._median.sel(x=slice(x0, x1), y=slice(y0, y1)).isnull().all():
                print("cell", chunk_name, "will be skipped. data is present")
            else:
                node_list.append(chunk_name)
    with h5py.File(cache_fullname, "r") as h5:
        if l2_type == "swath":
            h5["swath"].visititems(collect_chunk_names)
        elif l2_type == "poca":
            h5["poca"].visititems(collect_chunk_names)
        elif l2_type in ["all", "both"]:
            Exception("Joined swath and poca aggregation is not completely implemented.")
            h5.visititems(collect_chunk_names)
    print("processing queue contains:\n", node_list)
    print("\nGridding the data. Each chunk at a time...")
    # for the loop below, multiprocessing could be used. however, the
    # implementation should save intermediate results if interupted.
    for chunk_name in node_list:
        print("-----\n\nnext chunk:", chunk_name)
        with h5py.File(cache_fullname, "r") as h5:
            period_list = list(h5["/".join(["swath"] + chunk_name)].keys())
        l2_df = pd.concat([pd.read_hdf(cache_fullname, "/".join(["swath"] + chunk_name + [period]), mode="r", ) for period in sorted(period_list)], axis=0)
        # one could drop some of the data before gridding. however, excluding
        # off-glacier data is expensive and filtering large differences to the
        # DEM can hide issues while statistics like the median and the IQR
        # should be fairly robust.
        if len(l2_df.index) != 0:
            l2_df = l2_df.loc[ext_t_axis[0]:ext_t_axis[-1]]
            l2_df[["x", "y"]] = ((l2_df[["x", "y"]]//spatial_res_meter+.5)*spatial_res_meter).astype("i4")
            l2_df["roll_0"] = pd.cut(l2_df.index, bins=ext_t_axis, right=False, labels=False, include_lowest=True)
            # note on the for-loops:
            #     because of the late-binding python behavior, one or the other way the
            #     counting index must be defined at place (as opposed to when dask tries
            #     to calculate the values because then all the indeces have the same
            #     value). the chosen way is defining a function which creates a new
            #     namespace (in which the index is copied and will not be changed from
            #     outside).
            for i in range(1, window_ntimesteps):
                l2_df[f"roll_{i}"] = l2_df.roll_0-i
            for i in range(window_ntimesteps):
                l2_df[f"roll_{i}"] = l2_df[f"roll_{i}"].astype("i4")//window_ntimesteps
            results_list = [None]*window_ntimesteps
            for i in range(window_ntimesteps):
                def local_closure(roll_iteration):
                    # note: consider calculating the kurtosis of the data between the 25th
                    #       and the 75th percentile. this could help later on to identify
                    #       the approximate distribution shape
                    return l2_df.rename(columns={f"roll_{i}": "time_idx"}).groupby(["time_idx", "x", "y"]).h_diff.apply(agg_func_and_meta[0])
                results_list[i] = local_closure(i)
            del l2_df
            for i in range(window_ntimesteps):
                results_list[i] = results_list[i].droplevel(3, axis=0)
                results_list[i].index = results_list[i].index.set_levels(
                    (results_list[i].index.levels[0]*window_ntimesteps+i+1), level=0).rename("time", level=0)
            l3_data = pd.concat(results_list).sort_index().loc[(slice(0,len(ext_t_axis)-1),slice(None),slice(None)),:]
            for df in results_list:
                del df
            l3_data.index = l3_data.index.remove_unused_levels()
            l3_data.index = l3_data.index.set_levels(
                    ext_t_axis[l3_data.index.levels[0]].astype("datetime64[ns]"), level=0)
            l3_data = l3_data.sort_index()
            l3_data = l3_data.query(f"time >= '{start_datetime}' and time <= '{end_datetime}'")

            # note on "erratically" renaming the data below:
            #   the function consumes much memory. I tried to use xarrays close
            #   function where possible to reduce the memory footprint. however,
            #   does not seem to help. xarrays cache might be the problem. see
            #   issue #32.

            # merge with previous data if there are any
            if "previously_processed_l3" in locals():
                tmp = previously_processed_l3.to_dataframe().dropna(how="any")
                previously_processed_l3.close()
                previously_processed_l3 = pd.concat([tmp, l3_data], axis=0)
                del tmp
                print(l3_data.head())
                print("data count of combined data", len(previously_processed_l3.index))
                # there may not be duplicates. if there are, there is a bug
                duplicates = previously_processed_l3.index.duplicated(keep=False)
                if any(duplicates):
                    print(previously_processed_l3.index[duplicates])
                    print(previously_processed_l3.tail(30))
                    raise KeyError("Duplicates found in merging new with preexisting data.")
            else:
                previously_processed_l3 = l3_data
            del l3_data
            tmp = previously_processed_l3.to_xarray().sortby("x").sortby("y")
            del previously_processed_l3
            previously_processed_l3 = fill_missing_coords(tmp).rio.write_crs(crs)
            tmp.close()
            # save/backup result
            tmp_backup_path = os.path.join(tmp_path, f"{region_id}_l3")
            # try to write new data to file. if anything goes wrong, restore. is this sufficiently safe?
            try:
                if os.path.isfile(outfilepath):
                    shutil.move(outfilepath, tmp_backup_path)
                previously_processed_l3.to_netcdf(outfilepath)
            except Exception as err:
                shutil.move(tmp_backup_path, outfilepath)
                print("\n")
                warnings.warn(
                    "Failed to write to netcdf! Restored previous state. Printing error"
                    + "message below and continuing. Attempting to save to temporary file.")
                try:
                    safety_net_tmp_file_path = os.path.join(tmp_path, f"tmp_l3_state__{datetime.datetime.strftime('%dT%H%M%S')}.nc")
                    previously_processed_l3.to_netcdf(safety_net_tmp_file_path)
                except Exception as err_inner:
                    print("\n")
                    warnings.warn(
                        "Failed again to save safety! There likely is a problem with the data."
                        +" Rethrowing errors.")
                    raise
                else:
                    print("\n", "Managed to save state to", safety_net_tmp_file_path)
                    print("\n", "Original error is printed below.", str(err), "\n")
            else:
                print(datetime.datetime.now())
                print(f"processed and stored cell", chunk_name)
                if os.path.isfile(tmp_backup_path):
                    os.remove(tmp_backup_path)
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


def fill_missing_coords(l3_data, minx: int = 9e7, miny: int = 9e7,
                                 maxx: int = -9e7, maxy: int = -9e7
                        ) -> xr.Dataset:
    # previous version inspired by user9413641
    # https://stackoverflow.com/questions/68207994/fill-in-missing-index-positions-in-xarray-dataarray
    # ! resx, resy = [int(r) for r in l3_data.rio.resolution()]
    # don't use `rio.resolution()`: this assumes no holes which renders this function obsolete
    l3_data = l3_data.sortby("x").sortby("y") # ensure monotonix x and y
    resx, resy = [l3_data[k].diff(k).min().values.astype("int") for k in ["x", "y"]]
    minx, miny = int(minx+resx/2), int(miny+resy/2)
    maxx, maxy = int(maxx-resx/2), int(maxy-resy/2)
    if l3_data["x"].min().values < minx:
        minx = l3_data["x"].min().values.astype("int")
    else:
        minx = int(minx + (l3_data["x"].min().values - minx)%resx - resx)
    if l3_data["y"].min().values < miny:
        miny = l3_data["y"].min().values.astype("int")
    else:
        miny = int(miny + (l3_data["y"].min().values - miny)%resy - resy)
    if l3_data["x"].max().values > maxx:
        maxx = l3_data["x"].max().values.astype("int")
    else:
        maxx = int(maxx - (maxx - l3_data["x"].max().values)%resx + resx)
    if l3_data["y"].max().values > maxy:
        maxy = l3_data["y"].max().values.astype("int")
    else:
        maxy = int(maxy - (maxy - l3_data["y"].max().values)%resy + resy)
    coords = {"x": range(minx, maxx+1, resx), "y": range(miny, maxy+1, resy)}
    return l3_data.reindex(coords, fill_value=np.nan)
__all__.append("fill_missing_coords")


def fill_voids(ds: xr.Dataset,
               main_var: str,
               error: str,
               *,
               elev: str = "ref_elev",
               per: tuple[str] = ("basin", "basin_group"),
               basin_shapes: gpd.GeoDataFrame = None,
                outlier_limit: float = 5,
                outlier_replace: bool = False,
                outlier_iterations: int = 1,
               ) -> xr.Dataset:
    # mention memory footprint in docstring: reindexing leaks and takes a s**t ton of memory. roughly 5-10x l3_data size in total.
    if any([grouper not in ["basin", "basin_group"] for grouper in per]):
        raise NotImplementedError
    if basin_shapes is None:
        # figure out region. limited to o2 meanwhile
        print("... loading basin outlines")
        o2code = find_region_id(ds, scope="o2")
        basin_shapes = load_o2region(o2code, product="glaciers").to_crs(ds.rio.crs)
    else:
        basin_shapes = basin_shapes.to_crs(ds.rio.crs)
    # polygons will be repaired in later functions. it may be more
    # transparent to do it here.
    ds = fill_missing_coords(ds, *basin_shapes.total_bounds)
    if elev not in ds: # tbi: the ref elevs should always be loaded again after fill missing coords!
        print("... appending reference DEM to dataset")
        ds = append_elevation_reference(ds, ref_elev_name=elev)
    ds[elev] = ds[elev].rio.clip(basin_shapes.make_valid())
    ref_elev_da = ds[elev].copy()
    for grouper in per:
        res = []
        if grouper=="basin":
            if "basin_id" not in ds:
                print("... assigning basin ids to grid cells")
                ds = append_basin_id(ds, basin_shapes)
            print("... interpolating per basin")
            for label, group in (pbar := tqdm.tqdm(ds.groupby(ds.basin_id, squeeze=False))):
                pbar.set_description(f"... current basin id: {label:.0f}")
                res.append(interpolate_hypsometrically(group, main_var=main_var, elev=elev, error=error, outlier_replace=outlier_replace))
        elif grouper=="basin_group":
            if "group_id" not in ds:
                print("... assigning basin groups to grid cells")
                ds = append_basin_group(ds, basin_shapes)
            print("... interpolating per basin group")
            res = []
            for label, group in (pbar := tqdm.tqdm(ds.groupby(ds.group_id, squeeze=False))):
                pbar.set_description(f"... current group id: {label:.0f}")
                res.append(interpolate_hypsometrically(group, main_var=main_var, elev=elev, error=error, outlier_replace=False))
        ds = xr.concat(res, "stacked_x_y")
        for each in res:
            each.close()
        del res
        ds = ds.unstack("stacked_x_y")
        ds = ds.sortby("x").sortby("y")
        # reindexing fills gaps that were created by the grouping. if there were
        # gaps, the reference elevations need to be filled/resetted.
        ds = ds.reindex_like(ref_elev_da, method=None, copy=False)
        ds[elev] = ref_elev_da
    # if there are still missing data, temporally interpolate (gaps shorter than 1 year)
    if "time" in ds.dims:
        ds[main_var] = ds[main_var].interpolate_na(dim="time", method="linear", max_gap=pd.Timedelta(days=367))
    # if there are still missing data, interpolate region wide ("global hypsometric interpolation")
    ds = interpolate_hypsometrically(ds.rio.clip(basin_shapes.make_valid()).stack({"stacked_x_y": ["x", "y"]}).dropna("stacked_x_y", how="all"), main_var=main_var, elev=elev, error=error, outlier_replace=False)
    ds = fill_missing_coords(ds.unstack("stacked_x_y").sortby("x").sortby("y"))
    return ds
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
