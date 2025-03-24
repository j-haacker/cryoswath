"""Functions to aggregate point elevation estimates into a regular grid"""

__all__ = [
    "cache_l2_data",
    "med_iqr_cnt",
    "build_path",
    "build_dataset",
    "preallocate_zarr",
]

import dask.array
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
import warnings
import xarray as xr

from . import l2
from .misc import (
    data_path,
    dataframe_to_rioxr,
    filter_kwargs,
    find_region_id,
    load_cs_ground_tracks,
    load_glacier_outlines,
    sandbox_write_to,
    tmp_path
)
from .gis import buffer_4326_shp, ensure_pyproj_crs, find_planar_crs


# numba does not do help here easily. using the numpy functions is as fast as it gets.
def med_iqr_cnt(data):
    quartiles = np.quantile(data, [0.25, 0.5, 0.75])
    return pd.DataFrame(
        [[quartiles[1], quartiles[2] - quartiles[0], len(data)]],
        columns=["_median", "_iqr", "_count"],
    )


def cache_l2_data(
    region_of_interest: str | shapely.Polygon,
    start_datetime: str | pd.Timestamp,
    end_datetime: str | pd.Timestamp,
    *,
    buffer_region_by: float = None,
    max_elev_diff: float = 150,
    timestep_months: int = 1,
    window_ntimesteps: int = 3,
    cache_filename: str = None,
    cache_filename_extra: str = None,
    crs: CRS | int = None,
    reprocess: bool = False,
    **l2_from_id_kwargs,
) -> None:
    """
    Cache Level-2 (L2) data for a specified region and time period.

    This function processes and stores essential L2 data in an HDF5
    file, downloading and processing Level-1b (L1b) files if they are
    not available. It supports buffering the region and time period to
    ensure no data is missed.

    Parameters:
        region_of_interest (str | shapely.Polygon): The region to process,
            specified as a RGI region ID (string) or a custom shapely Polygon.
        start_datetime (str | pd.Timestamp): The start date for the data
            to be cached.
        end_datetime (str | pd.Timestamp): The end date for the data to be cached.
        buffer_region_by (float, optional): Buffer distance (in meters) to
            expand the region of interest. Defaults to 30,000 meters if not provided.
        max_elev_diff (float, optional): Maximum elevation difference to filter
            the data. Defaults to 150 meters.
        timestep_months (int, optional): Time step in months. Defaults to 1 month.
        window_ntimesteps (int, optional): Number of time steps for the rolling
            window data aggregation. Must be an odd number. Defaults to 3.
        cache_filename (str, optional): Custom filename for the cached data.
            Defaults to a name derived from the region ID.
        cache_filename_extra (str, optional): Additional string to append to
            the cache filename. Defaults to None.
        crs (CRS | int, optional): Coordinate reference system for the data.
            If None, a planar CRS is determined automatically. Defaults to None.
        reprocess (bool, optional): Whether to reprocess existing data.
            Defaults to False.
        **l2_from_id_kwargs: Additional keyword arguments passed to the
            `l2.from_id` function.

    Returns:
        None: The function saves the processed data to an HDF5 file and does
        not return any value.

    Raises:
        Warning: If the `window_ntimesteps` is not an odd number, it is adjusted
        and a warning is issued.
    """
    if window_ntimesteps % 2 - 1:
        old_window = window_ntimesteps
        window_ntimesteps = window_ntimesteps // 2 + 1
        warnings.warn(
            f"The window should be a uneven number of time steps. You asked for "
            f"{old_window}, but it has been changed to {window_ntimesteps}."
        )
    # ! end time step should be included.
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    # this function only makes sense for multiple months, so assume input
    # was on the month scale and set end_datetime to end of month
    end_datetime = (
        end_datetime.normalize() + pd.offsets.MonthBegin() - pd.Timedelta(1, "s")
    )
    if buffer_region_by is None:
        # buffer_by defaults to 30 km to not miss any tracks. Usually,
        # 10 km should do.
        buffer_region_by = 30_000
    time_buffer_months = (window_ntimesteps * timestep_months) // 2
    print(
        "Caching l2 data for",
        (
            "the region " + region_of_interest
            if isinstance(region_of_interest, str)
            else "a custom area"
        ),
        f"from {start_datetime} to {end_datetime}",
        f"+-{relativedelta(months=time_buffer_months)}.",
    )
    cs_tracks = load_cs_ground_tracks(
        region_of_interest,
        start_datetime,
        end_datetime,
        buffer_period_by=relativedelta(months=time_buffer_months),
        buffer_region_by=buffer_region_by,
    )
    print(
        "First and last available ground tracks are on",
        f"{cs_tracks.index[0]} and {cs_tracks.index[-1]}, respectively.,",
        f"{cs_tracks.shape[0]} tracks in total."
        "\n[note] Run update_cs_ground_tracks, optionally with `full=True` or",
        "`incremental=True`, if you local ground tracks store is not up to",
        "date. Consider pulling the latest version from the repository.",
    )

    # ! exclude data out of regions total_bounds in l2.from_id
    # (?possible/logically consistent?)
    print(
        "Storing the essential L2 data in hdf5, downloading and",
        "processing L1b files if not available...",
    )
    if isinstance(region_of_interest, str):
        region_id = region_of_interest
        region_of_interest = load_glacier_outlines(region_id, "glaciers")
    else:
        region_id = "_".join(
            [
                f"{region_of_interest.centroid.x:.0f}",
                f"{region_of_interest.centroid.y:.0f}",
            ]
        )
    if cache_filename is None:
        cache_filename = region_id
    if cache_filename_extra is not None:
        cache_filename += "_" + cache_filename_extra
    cache_fullname = os.path.join(tmp_path, cache_filename)
    if crs is None:
        crs = find_planar_crs(shp=region_of_interest)
    else:
        crs = ensure_pyproj_crs(crs)
    # cutting to actual glacier outlines takes very long. if needed,
    # implement multiprocessing.
    # bbox = gpd.GeoSeries(
    #     shapely.box(*gpd.GeoSeries(region_of_interest,
    #                 crs=4326).to_crs(crs).bounds.values[0]),
    #     crs=crs)
    # below tries to balance a large cache file with speed. it is not meant
    # to retain data in the suroundings - this is merely needed for the
    # implicit `simplify`` which would come at the cost of data if not
    # buffered
    bbox = gpd.GeoSeries(buffer_4326_shp(region_of_interest, 3_000), crs=4326).to_crs(
        crs
    )
    with sandbox_write_to(cache_fullname) as target:
        l2.from_id(
            cs_tracks.index,
            reprocess=reprocess,
            save_or_return="save",
            cache_fullname=target,
            crs=crs,
            bbox=bbox,
            max_elev_diff=max_elev_diff,
            **filter_kwargs(
                l2.from_id,
                l2_from_id_kwargs,
                blacklist=["cache", "max_elev_diff", "save_or_return", "reprocess"],
            ),
        )
    print(
        "Successfully finished caching for",
        (
            "the region " + region_of_interest
            if isinstance(region_of_interest, str)
            else "a custom area"
        ),
        f"from {start_datetime} to {end_datetime}",
        f"+-{relativedelta(months=time_buffer_months)}.",
    )


def preallocate_zarr(path, bbox, crs, time_index, data_vars) -> None:
    x_dummy = np.arange(
        (bbox.bounds[0] // 500 + 0.5) * 500, bbox.bounds[2], 500, dtype="i4"
    )
    y_dummy = np.arange(
        (bbox.bounds[1] // 500 + 0.5) * 500, bbox.bounds[3], 500, dtype="i4"
    )
    array_dummy = xr.DataArray(
        dask.array.full(
            shape=(len(time_index), len(x_dummy), len(y_dummy)),
            fill_value=np.nan,
            dtype="f4",
        ),
        coords={"time": time_index, "x": x_dummy, "y": y_dummy},
    )
    (
        xr.merge([array_dummy.rename(stat) for stat in data_vars])
        .rio.write_crs(crs)
        .to_zarr(path, compute=False)
    )


def build_dataset(
    region_of_interest: str | shapely.Polygon,
    start_datetime: str | pd.Timestamp,
    end_datetime: str | pd.Timestamp,
    *,
    l2_type: str = "swath",
    buffer_region_by: float = None,
    max_elev_diff: float = 150,
    timestep_months: int = 1,
    window_ntimesteps: int = 3,
    spatial_res_meter: float = 500,
    agg_func_and_meta: tuple[callable, dict] = (
        med_iqr_cnt,
        {"_median": "f8", "_iqr": "f8", "_count": "i8"},
    ),
    cache_filename: str = None,
    cache_filename_extra: str = None,
    crs: CRS | int = None,
    reprocess: bool = False,
    **l2_from_id_kwargs,
):
    # include in docstring: function footprint = 2x resulting ds + 2Gb (min. 5Gb)
    if window_ntimesteps % 2 - 1:
        old_window = window_ntimesteps
        window_ntimesteps = window_ntimesteps // 2 + 1
        warnings.warn(
            "The window should be a uneven number of time steps. You asked for "
            f"{old_window}, but it has been changed to {window_ntimesteps}."
        )
    # ! end time step should be included.
    start_datetime, end_datetime = pd.to_datetime([start_datetime, end_datetime])
    # this function only makes sense for multiple months, so assume input
    # was on the month scale and set end_datetime to end of month
    end_datetime = (
        end_datetime.normalize() + pd.offsets.MonthBegin() - pd.Timedelta(1, "s")
    )
    print(
        "Building a gridded dataset of elevation estimates for",
        (
            "the region " + region_of_interest
            if isinstance(region_of_interest, str)
            else "a custom area"
        ),
        f"from {start_datetime} to {end_datetime} every {timestep_months} months for",
        f"a rolling window of {window_ntimesteps} time steps.",
    )
    if buffer_region_by is None:
        # buffer_by defaults to 30 km to not miss any tracks. Usually,
        # 10 km should do.
        buffer_region_by = 30_000
    time_buffer_months = (window_ntimesteps * timestep_months) // 2
    cs_tracks = load_cs_ground_tracks(
        region_of_interest,
        start_datetime,
        end_datetime,
        buffer_period_by=relativedelta(months=time_buffer_months),
        buffer_region_by=buffer_region_by,
    )
    print(
        "First and last available ground tracks are on",
        f"{cs_tracks.index[0]} and {cs_tracks.index[-1]}, respectively.,",
        f"{cs_tracks.shape[0]} tracks in total."
        "\n[note] Run update_cs_ground_tracks, optionally with `full=True` or",
        "`incremental=True`, if you local ground tracks store is not up to",
        "date. Consider pulling the latest version from the repository.",
    )

    # ! exclude data out of regions total_bounds in l2.from_id
    # (?possible/logically consistent?)
    print(
        "Storing the essential L2 data in hdf5, downloading and",
        "processing L1b files if not available...",
    )
    if isinstance(region_of_interest, str):
        region_id = region_of_interest
        region_of_interest = load_glacier_outlines(region_id, "glaciers")
    else:
        region_id = "_".join(
            [
                f"{region_of_interest.centroid.x:.0f}",
                f"{region_of_interest.centroid.y:.0f}",
            ]
        )
    if cache_filename is None:
        cache_filename = region_id
    if cache_filename_extra is not None:
        cache_filename += "_" + cache_filename_extra
    cache_fullname = os.path.join(tmp_path, cache_filename)
    if crs is None:
        crs = find_planar_crs(shp=region_of_interest)
    else:
        crs = ensure_pyproj_crs(crs)
    # cutting to actual glacier outlines takes very long. if needed,
    # implement multiprocessing.
    # bbox = gpd.GeoSeries(
    #     shapely.box(*gpd.GeoSeries(region_of_interest,
    #                 crs=4326).to_crs(crs).bounds.values[0]),
    #     crs=crs)
    # below tries to balance a large cache file with speed. it is not meant
    # to retain data in the suroundings - this is merely needed for the
    # implicit `simplify` which would come at the cost of data if not
    # buffered
    region_of_interest = (
        gpd.GeoSeries(buffer_4326_shp(region_of_interest, 3_000), crs=4326)
        .to_crs(crs)
        .make_valid()
    )

    with sandbox_write_to(cache_fullname) as target:
        l2.from_id(
            cs_tracks.index,
            reprocess=reprocess,
            save_or_return="save",
            cache_fullname=target,
            crs=crs,
            bbox=region_of_interest,
            max_elev_diff=max_elev_diff,
            **filter_kwargs(
                l2.from_id,
                l2_from_id_kwargs,
                blacklist=["cache", "max_elev_diff", "save_or_return", "reprocess"],
            ),
        )
    ext_t_axis = pd.date_range(
        start_datetime - pd.DateOffset(months=time_buffer_months),
        end_datetime + pd.DateOffset(months=time_buffer_months),
        freq=f"{timestep_months}MS",
    )
    # strip GeoSeries-container -> shapely.Geometry
    region_of_interest = region_of_interest.iloc[0]
    outfilepath = build_path(region_id, timestep_months, spatial_res_meter)
    if reprocess and os.path.isdir(outfilepath):
        shutil.rmtree(outfilepath)
    if os.path.isdir(outfilepath):
        previously_processed_l3 = xr.open_zarr(outfilepath, decode_coords="all")
    else:
        preallocate_zarr(
            outfilepath,
            region_of_interest,
            crs,
            ext_t_axis,
            agg_func_and_meta[1].keys(),
        )
    ext_t_axis = ext_t_axis.astype("int64")
    node_list = []

    def collect_chunk_names(name, node):
        nonlocal node_list
        name_parts = name.split("/")
        if (
            not isinstance(node, h5py.Group)
            or len(name_parts) < 2
            or not name_parts[-2].startswith("x_")
        ):
            return
        chunk_name = name_parts[:2]
        if chunk_name not in node_list:
            x_range, y_range = chunk_name
            x0, x1 = [int(item) for item in x_range.split("_")[-2:]]
            y0, y1 = [int(item) for item in y_range.split("_")[-2:]]
            if (
                x0 > region_of_interest.bounds[2]
                or x1 < region_of_interest.bounds[0]
                or y0 > region_of_interest.bounds[3]
                or y1 < region_of_interest.bounds[1]
            ):
                return
            if not shapely.box(x0, y0, x1, y1).intersects(region_of_interest):
                print(
                    "cell",
                    chunk_name,
                    "will be skipped. cell does not intersect current region",
                )
            elif (
                "previously_processed_l3" in locals()
                and not previously_processed_l3._median.sel(
                    x=slice(x0, x1), y=slice(y0, y1)
                )
                .isnull()
                .all()
            ):
                print("cell", chunk_name, "will be skipped. data is present")
            else:
                node_list.append(chunk_name)

    with h5py.File(cache_fullname, "r") as h5:
        if l2_type == "swath":
            h5["swath"].visititems(collect_chunk_names)
        elif l2_type == "poca":
            h5["poca"].visititems(collect_chunk_names)
        elif l2_type in ["all", "both"]:
            Exception(
                "Joined swath and poca aggregation is not completely implemented."
            )
            h5.visititems(collect_chunk_names)
    print("processing queue contains:\n", node_list)
    print("\nGridding the data. Each chunk at a time...")
    # for the loop below, multiprocessing could be used. however, the
    # implementation should save intermediate results if interupted.
    for chunk_name in node_list:
        print("-----\n\nnext chunk:", chunk_name)
        with h5py.File(cache_fullname, "r") as h5:
            period_list = list(h5["/".join(["swath"] + chunk_name)].keys())
        l2_df = pd.concat(
            [
                pd.read_hdf(
                    cache_fullname,
                    "/".join(["swath"] + chunk_name + [period]),
                    mode="r",
                )
                for period in sorted(period_list)
            ],
            axis=0,
        )
        # one could drop some of the data before gridding. however, excluding
        # off-glacier data is expensive and filtering large differences to the
        # DEM can hide issues while statistics like the median and the IQR
        # should be fairly robust.
        if len(l2_df.index) != 0:
            l2_df = l2_df.loc[ext_t_axis[0] : ext_t_axis[-1]]
            l2_df[["x", "y"]] = (
                (l2_df[["x", "y"]] // spatial_res_meter + 0.5) * spatial_res_meter
            ).astype("i4")
            l2_df["roll_0"] = pd.cut(
                l2_df.index,
                bins=ext_t_axis,
                right=False,
                labels=False,
                include_lowest=True,
            )
            # note on the for-loops:
            #     because of the late-binding python behavior, one or the other way the
            #     counting index must be defined at place (as opposed to when dask tries
            #     to calculate the values because then all the indeces have the same
            #     value). the chosen way is defining a function which creates a new
            #     namespace (in which the index is copied and will not be changed from
            #     outside).
            for i in range(1, window_ntimesteps):
                l2_df[f"roll_{i}"] = l2_df.roll_0 - i
            for i in range(window_ntimesteps):
                l2_df[f"roll_{i}"] = (
                    l2_df[f"roll_{i}"].astype("i4") // window_ntimesteps
                )
            results_list = [None] * window_ntimesteps
            for i in range(window_ntimesteps):

                def local_closure(roll_iteration):
                    # note: consider calculating the kurtosis of the data between the
                    #       25th and the 75th percentile. this could help later on to
                    #       identify the approximate distribution shape
                    return (
                        l2_df.rename(columns={f"roll_{i}": "time_idx"})
                        .groupby(["time_idx", "x", "y"])
                        .h_diff.apply(agg_func_and_meta[0])
                    )

                results_list[i] = local_closure(i)
            del l2_df
            for i in range(window_ntimesteps):
                results_list[i] = results_list[i].droplevel(3, axis=0)
                results_list[i].index = (
                    results_list[i]
                    .index.set_levels(
                        (results_list[i].index.levels[0] * window_ntimesteps + i + 1),
                        level=0,
                    )
                    .rename("time", level=0)
                )
            l3_data = (
                pd.concat(results_list)
                .sort_index()
                .loc[(slice(0, len(ext_t_axis) - 1), slice(None), slice(None)), :]
            )
            for df in results_list:
                del df
            l3_data.index = l3_data.index.remove_unused_levels()
            l3_data.index = l3_data.index.set_levels(
                ext_t_axis[l3_data.index.levels[0]].astype("datetime64[ns]"), level=0
            )
            l3_data = l3_data.sort_index()
            l3_data = l3_data.query(
                f"time >= '{start_datetime}' and time <= '{end_datetime}'"
            )

            try:
                (
                    dataframe_to_rioxr(l3_data, crs)
                    .rio.clip([region_of_interest])
                    .drop_vars(["spatial_ref"])
                    .to_zarr(outfilepath, region="auto")
                )  # [["_median", "_iqr", "_count"]]
            except Exception as err:
                print("\n")
                warnings.warn(
                    "Failed to write to zarr! Attempting to dump current dataframe."
                )
                try:
                    safety_net_tmp_file_path = os.path.join(
                        tmp_path,
                        "__".join(
                            [
                                f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}",
                                "l3_dfdump",
                                f"region_{region_id}",
                                "_".join(chunk_name),
                            ]
                        )
                        + ".feather",
                    )
                    l3_data.to_feather(safety_net_tmp_file_path)
                except Exception as err_inner:
                    print("\n")
                    warnings.warn(
                        "Failed to do an emergency dump!" + " Rethrowing errors:"
                    )
                    raise err_inner
                else:
                    print(
                        "\n",
                        "Managed to dump current dataframe to",
                        safety_net_tmp_file_path,
                    )
                    print("\n", "Original error is printed below.", str(err), "\n")
            else:
                print(datetime.datetime.now())
                print("processed and stored cell", chunk_name)
                print(l3_data.head())
    print("\n\n+++++++++++++ successfully build dataset ++++++++++++++\n\n")
    return xr.open_zarr(outfilepath, decode_coords="all")


def build_path(
    region_of_interest, timestep_months, spatial_res_meter, aggregation_period=None
):
    # ! implement parsing aggregation period
    if not isinstance(region_of_interest, str):
        region_id = find_region_id(region_of_interest)
    else:
        region_id = region_of_interest
    if timestep_months != 1:
        timestep_str = str(timestep_months) + "-"
    else:
        timestep_str = ""
    timestep_str += "monthly"
    if spatial_res_meter == 1000:
        spatial_res_str = "1km"
    elif np.floor(spatial_res_meter / 1000) < 2:
        spatial_res_str = f"{spatial_res_meter}m"
    else:
        # if the trailing ".0" should be omitted, that needs to be implemented here
        spatial_res_str = f"{round(spatial_res_meter/1000, 1)}km"
    return os.path.join(
        data_path, "L3", "_".join([region_id, timestep_str, spatial_res_str + ".zarr"])
    )
