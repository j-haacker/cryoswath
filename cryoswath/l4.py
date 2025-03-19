"""Library of functions to produce level 4 (L4) data"""

__all__ = [
    "difference_to_reference_dem",
    "add_meta_to_default_finalized_l3",
    "append_basin_id",
    "append_basin_group",
    "append_elevation_reference",
    "elevation_trend_raster_from_l3",
    "fill_voids",
    "fill_l3_voids",
    "fit_trend",
    "fit_trend__seasons_removed",
    "timeseries_form_gridded",
    "trend_with_seasons",
    # "differential_change",
    # "relative_change",
]

from datetime import datetime
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pathlib import Path
import rasterio.warp
import rioxarray as rioxr
from statsmodels.tsa.seasonal import seasonal_decompose
import tqdm
from typing import Literal
import xarray as xr

from .misc import (
    discard_frontal_retreat_zone,
    effective_sample_size,
    fill_missing_coords,
    find_region_id,
    get_dem_reader,
    interpolate_hypsometrically,
    l3_path,
    l4_path,
    load_glacier_outlines,
    nanoseconds_per_year,
)
from . import misc, l3

# notes for future development of `differential_change` and
# `relative_change`:
# the current code exploits yearly orbit repeats. however, the orbit was
# changed in July, 2020 (search the internet for "cryo2ice"). thus, it
# would be better, to separate the data into the periods before and
# after the manoeuvre.
# further: the stability of the hypsometric void filling fit could be
# increased by evaluating a variety of relative surface elevation
# changes and averaging the results. E.g. the change from 2010-09-01 to
# 2015-09-01 and the change from 2010-09-01 to 2020-09-01 minus the
# change from 2015-09-01 to 2020-09-01 have to be the same. So the best
# would be to calculate all relative changes, then find all meaningful
# combinations, and derive a final product by averaging those
# combinations.


def add_meta_to_default_finalized_l3(
        outdir: str | Path,
        your_name: str = "n/a",
        your_institution: str = "n/a",
):
    """Adds meta data to and changes variable names of L3 dataset

    This function expects that :func:py:`pathlib.Path().stat().st_mtime`
    returns seconds since Unix epoch and only makes sense if you ran:

    .. code-block:: python

        cryoswath.l3.build_dataset(region_id)
        cryoswath.l4.fill_voids_in_l3(region_id)
    
    If you deviated from default values in the processing, verify
    the output of this function.

    Args:
        outdir (str | Path): Path to output directory.
        your_name (str, optional): Your name. Defaults to "n/a".
        your_institution (str, optional): Your institution. Defaults to "n/a".
    """

    def metadata():
        return {
            "elev_diff": {
                "orig_name": "_median",
                "standard_name": "land_ice_surface_height_above_reference",
                "long_name": "Surface elevation difference wrt. DEM",
                "units": "m",
                "_FillValue": np.nan,
                "dtype": "float32",
                "ancillary_variables":
                    "elev_diff_ref elev_diff_error elev_diff_obs_count elev_diff_interp_flag",
                "description":
                    "Glacier surface height in meter above a reference elevation. The "
                    "reference elevation is stored in variable \"elev_diff_ref\". The "
                    "sum of both gives the height above the WGS84 ellipsoid."
            },
            "elev_diff_ref": {
                "orig_name": "ref_elev",
                "standard_name": "land_height_reference_above_WGS84",
                "long_name": "Surface elevation reference",
                "units": "m",
                "_FillValue": np.nan,
                "dtype": "float32",
                "description":
                    "Reference surface height above the WGS84 ellipsoid. This variable is "
                    "derived from ArcticDEM v4.1 100 m mosaik by linear interpolation to "
                    "the current raster."
            },
            "elev_diff_error": {
                "orig_name": "_iqr",
                "standard_name": "land_ice_surface_height_above_reference standard_error",
                "long_name": "Standard deviation of height observations",
                "units": "m",
                "_FillValue": np.nan,
                "dtype": "float32",
                "description":
                    "Standard deviation of height observations. It is derived from "
                    "the interquartile range of those point height estimates that are "
                    "aggregated into the current raster cell."
            },
            "elev_diff_obs_count": {
                "orig_name": "_count",
                "standard_name": "land_ice_surface_height_above_reference number_of_observations",
                "long_name": "Number of observations",
                "units": "",
                "_FillValue": 0,
                "dtype": "uint16",
                "description":
                    "Number of point height estimates that are aggregated into the current "
                    "raster cell. Note: this is not the number of independent measurements."
            },
            "elev_diff_interp_flag": {
                "orig_name": "filled_flag",
                "standard_name": "quality_flag",
                "long_name": "Interpolation type",
                "units": "",
                "_FillValue": -1,
                "dtype": "int8",
                "flag_values": np.arange(-2, 7).astype("int8"),
                "flag_meanings": [
                    "failed",
                    "no data",
                    "observed",
                    "cell-based",
                    "basin-based",
                    "group-based",
                    "linear 1 year",
                    "subregion-wide",
                    "linear/const cell-based"
                ],
                "description":
                    "This flag indicates how voids were filled. -2: \"failed\" indicates "
                    "failure of the interpolation (should not occur), -1: \"no data\" outside "
                    "of glaciers (overridden by _FillValue), 0: \"observed\" were not filled, "
                    "1: \"cell-based\" use a season-aware linear trend model, 2: "
                    "\"basin-based\" use a hypsometric model per time step, 3: "
                    "\"group-based\" are like (2) but fit the model to all cells in a group "
                    "of basins, 4: \"linear 1 year\" linearly interpolates values within the "
                    "period of 1 year per cell, 5: \"subregion-wide\" does the same as (2) "
                    "and (3) using all cells per time step, 6: \"linear/const cell-based\" "
                    "fills all remaining gaps per cell with the temporally interpolated value "
                    "and with the nearest value at the start and end of the dataset."
            },
            "basin_id": {
                "orig_name": "basin_id",
                "standard_name": "drainage_basin_identifier",
                "long_name": "Drainage basin ID",
                "units": "",
                "_FillValue": 0,
                "dtype": "uint32",
                "description":
                    "Integer value used to identify Randolph Glacier Inventory (RGI) version "
                    "7.0 glaciers. The values are unique within first order RGI regions. "
                    "They are the same as used by RGI as last part of the RGI glacier "
                    "identifier."
            },
            "group_id": {
                "orig_name": "group_id",
                "standard_name": "drainage_basin_group_identifier",
                "long_name": "Group ID",
                "units": "",
                "_FillValue": 0,
                "dtype": "int32",
                "description":
                    "Integer value used to identify groups of Randolph Glacier Inventory "
                    "(RGI) version 7.0 glaciers. The values are composed of a sign from the "
                    "groups latitude, followed by the glacier terminus type according to RGI "
                    "v7.0 being either 0 for land terminating, 1 for tidewater glacier, or 9 "
                    "for not assigned. Further, the rounded absolute latitude and the rounded "
                    "longitude (0-360 E) are attached."
            },
            # "": {
            #     "orig_name": "",
            #     "standard_name": "",
            #     "long_name": "",
            #     "units": "",
            #     "description": ""
            # },
        }
    for file in Path(misc.l4_path).rglob("*__elev_diff_to_ref_at_monthly_intervals.nc"):
        date_time = str(pd.to_datetime(file.stat().st_mtime, unit="s"))
        o1name = misc.rgi_code_translator(file.name.split("-")[0])
        o2name = misc.rgi_code_translator(file.name.split("_")[0])
        outpath = (
            Path(outdir) / "__".join([
                "Glacier_surface_elevation",
                o1name.replace(" ", "_"),
                o2name.replace(" (", "-").replace(")", "").replace("/", "-").replace(" ", "_"),
                "monthly_500x500m.nc"
            ])
        )
        if outpath.exists():
            continue
        global_meta = {
            "Conventions": "CF-1.12",
            "title": f"Monthly, 500 x 500 m glacier surface elevations of {o2name}, {o1name}",
            "institution": your_institution,
            "source": "CryoSat-2 SARIn",
            "history": "\n".join([
                "CryoSat-2 SARIn ESA Baseline E L1b",
                f"{date_time}: {your_name} using cryoswath v2.1"
            ]),
            # "references": "",
            # "comment": "",
        }
        _metadata = metadata()
        with xr.open_dataset(file) as ds:
            out = (
                ds
                .drop_encoding()
                .drop_vars(["band", "cov_i", "cov_j"])
                .rename_vars({
                    v.pop("orig_name"): k
                    for k, v in _metadata.items()
                })
            )
        out.attrs = global_meta
        out = out.drop_vars([
            _var for _var in out.data_vars if _var not in _metadata
        ])
        for _var, attrs in _metadata.items():
            out[_var].attrs.update({**attrs})
        out.to_netcdf(
            outpath,
            encoding={
                _var: {
                    "dtype": out[_var].attrs.pop("dtype"),
                    "_FillValue": out[_var].attrs.pop("_FillValue"),
                    "zlib": True,
                    "complevel": 5
                } for _var in out.data_vars
            }
        )


def append_basin_id(
    ds: xr.DataArray | xr.Dataset,
    basin_gdf: gpd.GeoDataFrame = None,
) -> xr.Dataset:
    if basin_gdf is None:
        raise NotImplementedError("Automatic basin loading is not yet implemented.")
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    ds["basin_id"] = xr.DataArray(
        -1.0,  # should be float. is converted later anyway and if defined here,
        # _FillValue can be nan
        coords={k: v for k, v in ds.coords.items() if k in ["x", "y"]},
        dims=["x", "y"],
        attrs={"_FillValue": np.nan},
    )
    for i in range(len(basin_gdf)):
        try:
            subset = ds.basin_id.rio.clip(basin_gdf.iloc[[i]].make_valid())
        except rioxr.exceptions.NoDataInBounds:
            continue
        subset = xr.where(
            subset.isnull(),
            ds.basin_id.loc[dict(x=subset.x, y=subset.y)],
            float(basin_gdf.iloc[i].rgi_id.split("-")[-1]),
        )
        ds["basin_id"].loc[dict(x=subset.x, y=subset.y)] = subset
    ds["basin_id"] = xr.where(ds.basin_id == -1, ds.basin_id._FillValue, ds.basin_id)
    return ds


def append_basin_group(
    ds: xr.DataArray | xr.Dataset,
    basin_gdf: gpd.GeoDataFrame = None,
) -> xr.Dataset:
    if basin_gdf is None:
        raise NotImplementedError("Automatic basin loading is not yet implemented.")
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    ds["group_id"] = xr.DataArray(
        -1.0,  # should be float. is converted later anyway and if defined here,
        # _FillValue can be nan
        coords={k: v for k, v in ds.coords.items() if k in ["x", "y"]},
        dims=["x", "y"],
        attrs={"_FillValue": np.nan},
    )
    for basin_tt_group in basin_gdf.groupby("term_type"):
        # cut latitude into degree slices
        n_lat_bins = max(
            1, round(basin_tt_group[1].cenlat.max() - basin_tt_group[1].cenlat.min())
        )
        # below, `observed=True` to grant compatibility with future pandas versions.
        for basin_lat_group in basin_tt_group[1].groupby(
            pd.cut(basin_tt_group[1].cenlat, bins=n_lat_bins), observed=True
        ):
            # similarly, cut longitude
            n_lon_bins = max(
                1,
                round(
                    (basin_lat_group[1].cenlon.max() - basin_lat_group[1].cenlon.min())
                    * np.cos(np.deg2rad(basin_lat_group[0].mid))
                ),
            )
            for basin_lon_group in basin_lat_group[1].groupby(
                pd.cut(basin_lat_group[1].cenlon, bins=n_lon_bins), observed=True
            ):
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
                group_id = int(
                    f"{np.sign(lat)*term_type:.0f}{np.abs(lat):02.0f}{lon % 360:03.0f}"
                )
                mask = xr.where(
                    mask.isnull(), ds.group_id.loc[dict(x=mask.x, y=mask.y)], group_id
                )
                ds["group_id"].loc[dict(x=mask.x, y=mask.y)] = mask
    ds["group_id"] = xr.where(ds.group_id == -1, ds.group_id._FillValue, ds.group_id)
    return ds


def append_elevation_reference(
    geospatial_ds: xr.Dataset | xr.DataArray,
    ref_elev_name: str = "ref_elev",
    dem_file_name_or_path: str = None,
) -> xr.Dataset:
    if isinstance(geospatial_ds, xr.DataArray):
        geospatial_ds = geospatial_ds.to_dataset()
    # finding a latitude to determine the reference DEM like below may be prone to bugs
    with get_dem_reader(
        (geospatial_ds if dem_file_name_or_path is None else dem_file_name_or_path)
    ) as dem_reader:
        with rioxr.open_rasterio(dem_reader) as ref_dem:
            ref_dem = ref_dem.rio.clip_box(
                *geospatial_ds.rio.transform_bounds(ref_dem.rio.crs)
            ).squeeze()
            ref_dem = xr.where(
                ref_dem == ref_dem._FillValue, np.nan, ref_dem
            ).rio.write_crs(ref_dem.rio.crs)
            ref_dem.attrs.update({"_FillValue": np.nan})
    geospatial_ds[ref_elev_name] = xr.align(
        ref_dem.rio.reproject_match(
            geospatial_ds,
            resampling=rasterio.warp.Resampling.average,
            nodata=ref_dem._FillValue,
        ),
        geospatial_ds,
        join="right",
    )[0]
    geospatial_ds[ref_elev_name].attrs.update({"_FillValue": np.nan})
    return geospatial_ds


def fill_voids(
    ds: xr.Dataset,
    main_var: str,
    error: str,
    *,
    elev: str = "ref_elev",
    per: tuple[str] = ("basin", "basin_group"),
    basin_shapes: gpd.GeoDataFrame = None,
    outlier_limit: float = 5,
    outlier_replace: bool = False,
    outlier_iterations: int = 1,
    fit_sanity_check: dict = None,
    filled_flag: str = None,
) -> xr.Dataset:
    # mention memory footprint in docstring: reindexing leaks and takes a s**t ton of
    # memory. roughly 5-10x l3_data size in total.
    if any([grouper not in ["basin", "basin_group"] for grouper in per]):
        raise NotImplementedError
    if basin_shapes is None:
        # figure out region. limited to o2 meanwhile
        print("... loading basin outlines")
        o2code = find_region_id(ds, scope="o2")
        basin_shapes = load_glacier_outlines(o2code, product="glaciers").to_crs(ds.rio.crs)
    else:
        basin_shapes = basin_shapes.to_crs(ds.rio.crs)
    # remove time steps without any data
    if "time" in ds.dims:
        ds = ds.dropna("time", how="all", subset=[main_var])
    # polygons will be repaired in later functions. it may be more
    # transparent to do it here.
    ds = fill_missing_coords(ds, *basin_shapes.total_bounds)
    if (
        elev not in ds
    ):  # tbi: the ref elevs should always be loaded again after fill missing coords!
        print("... appending reference DEM to dataset")
        ds = append_elevation_reference(ds, ref_elev_name=elev)
    ds[elev] = ds[elev].rio.clip(basin_shapes.make_valid())
    ref_elev_da = ds[elev].copy()
    for grouper in per:
        res = []
        if grouper == "basin":
            if "basin_id" not in ds:
                print("... assigning basin ids to grid cells")
                ds = append_basin_id(ds, basin_shapes)
            print("... interpolating per basin")
            for label, group in (
                pbar := tqdm.tqdm(
                    ds.groupby(ds.basin_id.where(~ds[elev].isnull()), squeeze=False)
                )
            ):
                pbar.set_description(f"... current basin id: {label:.0f}")
                if (
                    "time" in group
                    and (~group[main_var].isnull()).any("time").sum() > 100
                ) or (~group[main_var].isnull()).sum() > 100:
                    group = discard_frontal_retreat_zone(
                        group, "basin_id", main_var, elev
                    )
                res.append(
                    interpolate_hypsometrically(
                        group,
                        main_var=main_var,
                        elev=elev,
                        error=error,
                        outlier_replace=outlier_replace,
                        outlier_limit=outlier_limit,
                        fit_sanity_check=fit_sanity_check,
                        fill_flag=(
                            None if filled_flag is None else (filled_flag, 2)
                        ),
                    )
                )
        elif grouper == "basin_group":
            if "group_id" not in ds:
                print("... assigning basin groups to grid cells")
                ds = append_basin_group(ds, basin_shapes)
                if "basin_id" in ds:
                    ds["group_id"] = xr.where(ds.basin_id.isnull(), np.nan, ds.group_id)
            print("... interpolating per basin group")
            for label, group in (
                pbar := tqdm.tqdm(
                    ds.groupby(ds.group_id.where(~ds[elev].isnull()), squeeze=False)
                )
            ):
                pbar.set_description(f"... current group id: {label:.0f}")
                res.append(
                    interpolate_hypsometrically(
                        group,
                        main_var=main_var,
                        elev=elev,
                        error=error,
                        outlier_replace=False,
                        outlier_limit=outlier_limit,
                        fit_sanity_check=fit_sanity_check,
                        fill_flag=(
                            None if filled_flag is None else (filled_flag, 3)
                        ),
                    )
                )
        ds = xr.concat(res, "stacked_x_y")
        for each in res:
            each.close()
        del res
        ds = ds.unstack(
            "stacked_x_y",
            fill_value={
                _var: (
                    ds[_var].attrs["_FillValue"]
                    if "_FillValue" in ds[_var].attrs
                    else np.nan
                )
                for _var in ds.data_vars
            }
        )
        ds = ds.sortby("x").sortby("y")
        # reindexing fills gaps that were created by the grouping. if there were
        # gaps, the reference elevations need to be filled/resetted.
        ds = ds.reindex_like(
            ref_elev_da,
            method=None,
            copy=False,
            fill_value={
                _var: (
                    ds[_var].attrs["_FillValue"]
                    if "_FillValue" in ds[_var].attrs
                    else np.nan
                )
                for _var in ds.data_vars
            }
        )
        ds[elev] = ref_elev_da
    # if there are still missing data, temporally interpolate (gaps shorter than 1 year)
    if "time" in ds.dims:
        _new_main = ds[main_var].interpolate_na(
            dim="time", method="linear", max_gap=pd.Timedelta(days=367)
        )
        _interpolated = np.logical_and(ds[main_var].isnull(), ~_new_main.isnull())
        ds[filled_flag] = xr.where(
            ~_interpolated,
            ds[filled_flag],
            4,
            keep_attrs=True
        )
        ds[error] = ds[error].interpolate_na(
            dim="time", method="nearest", max_gap=pd.Timedelta(days=367)
        )
        ds[main_var] = _new_main
    # if there are still missing data, interpolate region wide ("global
    # hypsometric interpolation")
    ds = interpolate_hypsometrically(
        (ds.where(~ds.basin_id.isnull(), xr.Dataset({
                _var: (
                    ds[_var].attrs["_FillValue"]
                    if "_FillValue" in ds[_var].attrs
                    else np.nan
                )
                for _var in ds.data_vars
            })) if "basin_id" in ds else ds)
        .rio.clip(basin_shapes.make_valid())
        .stack({"stacked_x_y": ["x", "y"]})
        .dropna("stacked_x_y", how="any", subset=[elev]),
        main_var=main_var,
        elev=elev,
        error=error,
        outlier_replace=False,
        outlier_limit=outlier_limit,
        fit_sanity_check=fit_sanity_check,
        fill_flag=(
            None if filled_flag is None else (filled_flag, 5)
        ),
    )
    # if there are STILL missing data, temporally interpolate remaining
    # gaps and fill the margins. this should only occur at first and
    # last time steps
    if "time" in ds.dims:
        _void = ds[main_var].isnull()
        ds[main_var] = (
            ds[main_var]
            .interpolate_na(dim="time", method="linear")
            .bfill("time")
            .ffill("time")
        )
        _interpolated = np.logical_and(~ds[main_var].isnull(), _void)
        ds[error] = xr.where(
            ~_interpolated,
            ds[error],
            50
        )
        ds[filled_flag] = xr.where(
            ~_interpolated,
            ds[filled_flag],
            6,
            keep_attrs=True
        )
    ds = fill_missing_coords(
        ds.unstack(
            "stacked_x_y",
            fill_value={
                _var: (
                    ds[_var].attrs["_FillValue"]
                    if "_FillValue" in ds[_var].attrs
                    else np.nan
                )
                for _var in ds.data_vars
            }
        )
    )
    return ds


def fill_l3_voids(o2region: str) -> xr.Dataset:
    """Fills data gaps in L3 datasets

    This is a convenience function that currently takes opinionated
    choices that cannot be modified. More capabilities are planned to be
    implemented in future.

    This function, also, writes the results to disk. If a file is
    present at the corresponding location, it is read instead of being
    recalculated.

    Args:
        o2region (str): The RGI second order region identifier.

    Returns:
        xr.Dataset: Filled dataset.
    """

    results_path = os.path.join(
        l4_path,
        o2region + "__elev_diff_to_ref_at_monthly_intervals.nc"
    )

    if os.path.isfile(results_path):
        return xr.open_dataset(results_path, decode_coords="all")

    basins_gdf = misc.load_glacier_outlines(o2region, product="glaciers", union=False)
    if o2region in ["08-01", "08-02", "08-03"]:
        o2region = "08_scandinavia"
    ds = xr.open_zarr(os.path.join(misc.l3_path, o2region+"_monthly_500m.zarr"), decode_coords="all").load()
    crs = ds.rio.crs
    expected_fit_results_path = os.path.join(
        misc.l4_path,
        f"surface_elevation_trend__rgi-o2region_{o2region}.zarr"
    )
    if os.path.isdir(expected_fit_results_path):
        fit_rm_outl_res = xr.open_zarr(expected_fit_results_path, decode_coords="all").load()
    else:
        fit_rm_outl_res = elevation_trend_raster_from_l3(o2region, only_intermediate=True)
    if o2region == "08_scandinavia":
        ds = ds.rio.clip_box(*basins_gdf.to_crs(crs).total_bounds)
        fit_rm_outl_res = fit_rm_outl_res.rio.clip_box(*basins_gdf.to_crs(crs).total_bounds)
    ds = ds.where(ds._count > 3).where(ds._iqr < 30).dropna("time", how="all")
    ds["_iqr"] = xr.where(ds._median.isnull(),
                        fit_rm_outl_res.RMSE * 2 * misc._norm_isf_25,
                        ds._iqr)
    ds["filled_flag"] = xr.where(ds._median.isnull(), 1, 0).astype("i1")
    ds.filled_flag.attrs["_FillValue"] = -1
    ds["_median"] = ds._median.fillna(trend_with_seasons(
        ds.time.astype("int"),
        **{
            param: fit_rm_outl_res.curvefit_coefficients.sel(param=param)
            for param in fit_rm_outl_res.param.values
        }
    ))
    mask = np.logical_and(
        fit_rm_outl_res.curvefit_covariance.sel(cov_i="trend", cov_j="trend") < 2,
        np.logical_and(fit_rm_outl_res.curvefit_covariance.sel(cov_i="amp_yearly", cov_j="amp_yearly") < 100,
                    fit_rm_outl_res.curvefit_covariance.sel(cov_i="amp_semiyr", cov_j="amp_semiyr") < 100)
    )
    ds["_median"] = xr.where(mask, ds._median, np.nan)
    ds["filled_flag"] = xr.where(mask, ds.filled_flag, -2, keep_attrs=True)  # should not survive void filling!
    ds["_iqr"] = ds._iqr.where(~ds._median.isnull())
    ds = misc.fill_missing_coords(ds.rio.write_crs(crs), *basins_gdf.to_crs(crs).total_bounds)
    ds = ds.rio.clip(basins_gdf.to_crs(crs).make_valid())
    return difference_to_reference_dem(ds, save_to_disk=results_path, basin_shapes=basins_gdf)


def fit_trend(
    data: xr.Dataset,
    *,
    pivot: pd.DateOffset,  # ? best class?
    timestep_months: int = 12,
    return_raw: bool = False,
) -> xr.Dataset:
    # using resample(time="...").nearest(pd.Timedelta(..., "days"))\
    #   .dropna("time", "all")
    # it could theoretically be implemented to select a valid value in the
    # proximity of the desired time stamp. because the required frequency is
    # difficult to define flexibly and for the benefit of a well-defined
    # time stamp, a different approach is taken. For a longer, e.g.,
    # 3-monthly, aggregation time, the current approach should work equally
    # fine.
    time_indices = pd.date_range(
        data.time[0].values + pivot, data.time[-1].values, freq=f"{timestep_months}MS"
    )
    data = data.sel(time=time_indices)
    data = data.where(data.isel(time=slice(None, 3)).any("time")).where(
        data.isel(time=slice(-3, None)).any("time")
    )
    fit_res = data.polyfit("time", 1, cov=True)
    fit_res["polyfit_coefficients"][0] = (
        fit_res["polyfit_coefficients"][0] * nanoseconds_per_year
    )
    fit_res["polyfit_covariance"][0, 0] = (
        fit_res["polyfit_covariance"][0, 0] * nanoseconds_per_year**2
    )
    fit_res["polyfit_covariance"][0, 1] = (
        fit_res["polyfit_covariance"][0, 1] * nanoseconds_per_year
    )
    fit_res["polyfit_covariance"][1, 0] = (
        fit_res["polyfit_covariance"][1, 0] * nanoseconds_per_year
    )
    if return_raw:
        return fit_res
    ds = xr.Dataset()
    ds["trend"] = fit_res.polyfit_coefficients.sel(degree=1)
    ds["trend_CI95"] = 2 * fit_res.polyfit_covariance.isel(cov_i=0, cov_j=0) ** 0.5

    def trunc_weights(CI, trend):
        if CI < np.min([1.5, 0.5 + 0.2 * np.abs(trend)]):
            return 2 / CI
        else:
            return 0

    ds["weights"] = xr.apply_ufunc(
        trunc_weights, ds.trend_CI95, ds.trend, vectorize=True
    )
    return fill_voids(ds.rio.write_crs(data.rio.crs), "trend", "trend_CI95")


def difference_to_reference_dem(
    l3_data: xr.Dataset,
    save_to_disk: str | bool = True,
    basin_shapes: gpd.GeoDataFrame = None,
) -> xr.Dataset:
    if (np.abs(l3_data._median) < 150).any():
        Exception("_median deviates more than 150 m from reference")
    for _var in ["_median", "_iqr", "_count"]:
        l3_data[_var] = l3_data[_var].astype("f4")
    res = fill_voids(
        l3_data,
        main_var="_median",
        error="_iqr",
        elev="ref_elev",
        basin_shapes=basin_shapes,
        per=("basin", "basin_group"),
        outlier_replace=False,
        fit_sanity_check=True,
        filled_flag="filled_flag",
    )
    # print(res.filled_flag.dtype, np.unique(res.filled_flag))
    if save_to_disk:
        try:
            region_id = find_region_id(l3_data)
        except Exception as err:
            import traceback

            print(traceback.format_exc())
            print(str(err))
            region_id = str(datetime.now())
        res.to_netcdf(  # .drop_encoding()
            os.path.join(
                misc.l4_path,
                (
                    save_to_disk
                    if isinstance(save_to_disk, str)
                    else region_id + "__elev_diff_to_ref_at_monthly_intervals.nc"
                ),
            ),
            # encoding={  doesn't do anything
            #     _var: {
            #         "dtype": res[_var].dtype,
            #         # "_FillValue": res[_var].attrs["_FillValue"],
            #     }
            #     for _var in res.data_vars
            #     if "_FillValue" in res[_var].attrs
            # }
        )
    return res


def elevation_trend_raster_from_l3(region_id: str, *, only_intermediate: bool = False) -> xr.Dataset:
    """Calculate elevation trend for each cell of L3 dataset

    This is a convenience function that currently takes opinionated
    choices that cannot be modified. More capabilities are planned to be
    implemented in future.

    Next to returning the elevation trends, these are, also, written to
    disk. If the result files are present, these will be read instead of
    repeating the calculation.

    Args:
        region_id (str): Region identifier that was used in the L3 Zarr
            store names.
        only_intermediate (bool, optional): If true, returns after
            calculating the trends where sufficient data is available
            and skips filling voids. Defaults to False.

    Returns:
        xr.Dataset: Dataset of elevation trends and other parameters,
        depending on the usage.
    """

    # paths of (intermediate) results
    filename = f"surface_elevation_trend__rgi-o2region_{region_id}"
    interm_res_path = os.path.join(
        l4_path,
        filename + ".zarr"
    )
    result_path = os.path.join(
        l4_path,
        filename + "__m_yr-1.tif"
    )

    # first part: fitting trends
    if not os.path.isdir(interm_res_path):
        ds = xr.open_zarr(os.path.join(l3_path, region_id+"_monthly_500m.zarr"), decode_coords="all")#.load()
        ds = ds.where(ds._count>3).where(ds._iqr<30)
        ds = ds.where(np.logical_and((~ds._median.isel(time=slice(None, 30)).isnull()).sum("time")>5, (~ds._median.isel(time=slice(-30, None)).isnull()).sum("time")>5))
        ds = ds.chunk(dict(time=-1))
        fit_res = ds._median.transpose('time', 'y', 'x').curvefit(
            coords="time",
            func=trend_with_seasons,
            param_names=["trend", "offset", "amp_yearly", "phase_yearly", "amp_semiyr", "phase_semiyr"],
            bounds={"amp_yearly": (0, np.inf),
                    "phase_yearly": [-np.pi, np.pi],
                    "amp_semiyr": (0, np.inf),
                    "phase_semiyr": [-np.pi, np.pi]},
            errors="ignore"
        )
        # # debugging output:
        # fit_res.curvefit_coefficients.sel(param="trend").rio.write_crs(ds.rio.crs).rio.to_raster("../figures/source_data/new_outl_still_present_surface_elevation_trend__rgi-o2region_"
        #                 + f"{o1:02d}-{o2:02d}__m_yr-1.tif")
        model_vals = xr.apply_ufunc(trend_with_seasons,
                                    ds.time.astype("int"),
                                    *[fit_res.sel(param=p) for p in fit_res.param],
                                    dask="allowed")
        residuals = ds._median - model_vals.rename({"curvefit_coefficients": "_median"})._median
        res_std = residuals.std("time")
        outlier = np.abs(residuals)-2*res_std>0
        fit_rm_outl_res = ds._median.where(~outlier).transpose('time', 'y', 'x').curvefit(
            coords="time",
            func=trend_with_seasons,
            param_names=["trend", "offset", "amp_yearly", "phase_yearly", "amp_semiyr", "phase_semiyr"],
            bounds={"amp_yearly": (0, np.inf),
                    "phase_yearly": [-np.pi, np.pi],
                    "amp_semiyr": (0, np.inf),
                    "phase_semiyr": [-np.pi, np.pi]},
            errors="ignore"
        ).rio.write_crs(ds.rio.crs)
        model_vals = xr.apply_ufunc(trend_with_seasons,
                                    ds.time.astype("int"),
                                    *[fit_rm_outl_res.sel(param=p) for p in fit_rm_outl_res.param],
                                    dask="allowed")
        residuals = ds._median - model_vals.rename({"curvefit_coefficients": "_median"})._median
        fit_rm_outl_res["RMSE"] = (residuals**2).mean("time")**.5
        fit_rm_outl_res.to_zarr(interm_res_path, mode="w")
    else:
        fit_rm_outl_res = xr.open_zarr(
            interm_res_path,
            decode_coords="all"
        ).load()
    if only_intermediate:
        return fit_rm_outl_res

    # second part: filling voids
    if not os.path.isfile(result_path):
        basin_gdf = load_glacier_outlines(region_id, "basins", False)
        mask = np.logical_and(
            fit_rm_outl_res.curvefit_covariance.sel(cov_i="trend", cov_j="trend") < 2,
            np.logical_and(fit_rm_outl_res.curvefit_covariance.sel(cov_i="amp_yearly", cov_j="amp_yearly") < 100,
                            fit_rm_outl_res.curvefit_covariance.sel(cov_i="amp_semiyr", cov_j="amp_semiyr") < 100)
        )
        fit_rm_outl_res = fit_rm_outl_res.where(mask).sel(param="trend", cov_i="trend", cov_j="trend")
        fit_rm_outl_res["trend"] = fit_rm_outl_res.curvefit_coefficients
        fit_rm_outl_res["trend_std"] = fit_rm_outl_res.curvefit_covariance**.5
        filled = fill_voids(fit_rm_outl_res[["trend", "trend_std"]], "trend", "trend_std", basin_shapes=basin_gdf, outlier_replace=True, outlier_limit=2)
        # give rasterio a hint about the nodata values
        filled.trend.attrs["_FillValue"] = np.nan
        filled.trend_std.attrs["_FillValue"] = np.nan
        # print(filled.trend.attrs, filled.trend_std.attrs)
        filled[["trend", "trend_std"]].transpose("y", "x").rio.to_raster(
            result_path
        )
    else:
        filled = rioxr.open_rasterio(result_path)
    return filled


def fit_trend__seasons_removed(l3_ds: xr.Dataset) -> xr.Dataset:
    l3_ds = l3_ds.where(
        np.logical_and(
            (~l3_ds._median.isel(time=slice(None, 30)).isnull()).sum("time") > 5,
            (~l3_ds._median.isel(time=slice(-30, None)).isnull()).sum("time") > 5,
        )
    )
    if "chunks" in l3_ds._median:
        l3_ds = l3_ds.chunk(dict(time=-1))
    fit_res = l3_ds._median.transpose("time", "y", "x").curvefit(
        coords="time",
        func=trend_with_seasons,
        param_names=[
            "trend",
            "offset",
            "amp_yearly",
            "phase_yearly",
            "amp_semiyr",
            "phase_semiyr",
        ],
        bounds={
            "amp_yearly": (0, np.inf),
            "phase_yearly": [-np.pi, np.pi],
            "amp_semiyr": (0, np.inf),
            "phase_semiyr": [-np.pi, np.pi],
        },
        errors="ignore",
    )
    model_vals = xr.apply_ufunc(
        trend_with_seasons,
        l3_ds.time.astype("int"),
        *[fit_res.sel(param=p) for p in fit_res.param],
        dask="allowed",
    )
    residuals = (
        l3_ds._median - model_vals.rename({"curvefit_coefficients": "_median"})._median
    )
    res_std = residuals.std("time")
    outlier = np.abs(residuals) - 2 * res_std > 0
    fit_rm_outl_res = (
        l3_ds._median.where(~outlier)
        .transpose("time", "y", "x")
        .curvefit(
            coords="time",
            func=trend_with_seasons,
            param_names=[
                "trend",
                "offset",
                "amp_yearly",
                "phase_yearly",
                "amp_semiyr",
                "phase_semiyr",
            ],
            bounds={
                "amp_yearly": (0, np.inf),
                "phase_yearly": [-np.pi, np.pi],
                "amp_semiyr": (0, np.inf),
                "phase_semiyr": [-np.pi, np.pi],
            },
            errors="ignore",
        )
        .rio.write_crs(l3_ds.rio.crs)
    )
    model_vals = xr.apply_ufunc(
        trend_with_seasons,
        l3_ds.time.astype("int"),
        *[fit_rm_outl_res.sel(param=p) for p in fit_rm_outl_res.param],
        dask="allowed",
    )
    residuals = (
        l3_ds._median - model_vals.rename({"curvefit_coefficients": "_median"})._median
    )
    fit_rm_outl_res["RMSE"] = (residuals**2).mean("time") ** 0.5
    return fit_rm_outl_res.rio.write_crs(l3_ds.rio.crs)


def differential_change(
    data: xr.Dataset,
    save_to_disk: str | bool = True,
) -> xr.Dataset:
    # ! needs to be tested again

    # roughly filter data
    data = data.where(data._count > 3).where(np.abs(data._median) < 150)
    # the `shift` below sets back the data 1 year, such that this translates
    # to "later minus earlier"
    shiftby = np.argwhere(
        (data.time == data.time[0].values + pd.DateOffset(years=1)).values
    )[0][0]
    differences = (data._median - data._median.shift(time=shiftby)).dropna(
        "time", how="all"
    )
    uncertainties = (data._iqr**2 + data._iqr.shift(time=shiftby) ** 2).dropna(
        "time", how="all"
    ) ** 0.5
    if "ref_elev" not in data:
        data = l3.append_elevation_reference(data, ref_elev_name="ref_elev")
    data = xr.merge(
        [
            differences.rename("elev_change"),
            uncertainties.rename("elev_change_CI95"),
            data.ref_elev,
        ]
    )
    res = fill_voids(
        data,
        "elev_change",
        error="elev_change_CI95",
        elev="ref_elev",
        per=("basin", "basin_group"),
        outlier_limit=2,
        outlier_replace=True,
        outlier_iterations=3,
    )
    if save_to_disk:
        res.to_netcdf(
            os.path.join(
                misc.l4_path,
                (
                    save_to_disk
                    if isinstance(save_to_disk, str)
                    else find_region_id(data)
                    + "__yearly_changes_at_monthly_intervals.nc"
                ),
            )
        )
    return res


def relative_change(
    l3_data: xr.Dataset,
    basin_shapes: gpd.GeoDataFrame = None,
    glac_ref_year: int = 2010,
    pivot_month: int = 9,
    save_to_disk: str | bool = True,
) -> xr.Dataset:
    # ! needs to be tested

    if isinstance(basin_shapes, str):
        basin_shapes = gpd.GeoSeries(
            load_glacier_outlines(basin_shapes, "glaciers", False)
        )
    if glac_ref_year < 2010:
        glac_ref_year += 2000
    # roughly filter data
    l3_data = l3_data.where(l3_data._count > 3).where(np.abs(l3_data._median) < 150)
    ref_period = pd.date_range(
        f"{glac_ref_year}-{pivot_month:02d}", freq="MS", periods=12
    )
    reference = xr.Dataset(
        dict(
            _median=(("month", "x", "y"), l3_data._median.sel(time=ref_period).values),
            _iqr=(("month", "x", "y"), l3_data._iqr.sel(time=ref_period).values),
        ),
        coords={"month": ref_period.month, "x": l3_data.x, "y": l3_data.y},
    )
    values = (
        l3_data._median - reference._median.sel(month=l3_data.time.dt.month)
    ).drop("month")
    uncertainties = (
        l3_data._iqr**2 + reference._iqr.sel(month=l3_data.time.dt.month) ** 2
    ).drop("month") ** 0.5
    if "ref_elev" not in l3_data:
        l3_data = l3.append_elevation_reference(l3_data, ref_elev_name="ref_elev")
    l3_data = xr.merge(
        [values.rename("elevation"), uncertainties.rename("error"), l3_data.ref_elev]
    )
    res = fill_voids(
        l3_data.drop_sel(time=ref_period),
        main_var="elevation",
        error="error",
        elev="ref_elev",
        basin_shapes=basin_shapes,
        per=("basin", "basin_group"),
        outlier_limit=2,
        outlier_replace=True,
        outlier_iterations=3,
    )
    l3_data["error"] = xr.where(l3_data.time.isin(ref_period), 0, l3_data.error)
    res = xr.merge(
        [res, l3_data.sel(time=ref_period).fillna(0)], join="outer", compat="override"
    )
    if save_to_disk:
        res.to_netcdf(
            os.path.join(
                misc.l4_path,
                (
                    save_to_disk
                    if isinstance(save_to_disk, str)
                    else find_region_id(l3_data)
                    + "__relative_elevation_estimates_at_monthly_intervals.nc"
                ),
            )
        )
    return res


def timeseries_from_gridded(
        ds: xr.Dataset,
        void_err_type: Literal["confidence", "prediction"] = "prediction",
) -> pd.DataFrame:
    """Calculates uncertainties of average elevation

    This function returns a two-column DataFrame of elevation averages
    and the associated uncertaities. Computing the uncertainties is not
    trivial and will be reasoned in an upcoming paper.

    The void filling flags are essential. Currently they are hard-coded
    and have the following meaning:
    -2: filling failed
    -1/nan: no data
    0: no filling
    1: filled based on :func:`trend_with_season` fit per cell
    2: :func:`misc.interpolate_hypsometrically` per basin
    3: :func:`misc.interpolate_hypsometrically` per group of basins
    4: linear temporal interpolation for gaps shorter than one year
    5: second order region wide :func:`misc.interpolate_hypsometrically`
    6: filling with temporally closest value per cell

    Select `void_err_type` depending on whether prediction or confidence
    intervals were assigned. Note that in any case the errors are
    expected to be on the scale of interquartile ranges, i.e., the
    errors will be multiplied by a scaling factor.

    Args:
        ds (xr.Dataset): Void-filled L3 dataset. Needs variables
            "_median", "_iqr", "basin_id", "group_id, "filled_flag",
            "x", "y", and "time".
        void_err_type (Literal["prediction", "confidence"]): Type of
            errors for filled voids. Defaults to "prediction".

    Returns:
        pd.DataFrame: DataFrame with columns "elevation" and
            "uncertainty", and time stamps on the index.
    """
    decmp_res = seasonal_decompose(ds._median.mean(["x", "y"]), period=12, extrapolate_trend=True)
    results = pd.DataFrame(columns=["elevation", "uncertainty"])
    results["elevation"] = ds._median.mean(["x", "y"]).to_series() - decmp_res.trend[0]
    
    # add large error value where no observations are available at all
    ds["_iqr"] = xr.where(
        ds.filled_flag != -2,
        ds._iqr,
        50
    )

    if void_err_type == "confidence":
        da = ds._iqr.where(ds.filled_flag.isin([0, 1]))
        _coarsened = da.coarsen(
            {"x": 4, "y": 4},
            boundary="pad",
        )
        _weights = _coarsened.count()
        num_cells = _weights.sum(["x", "y"])
        unc1 = (
            ((_coarsened.mean() * _weights) ** 2).sum(["x", "y"]) ** 0.5
            # / _weights.sum(["x", "y"]) <- for the current weighting
            # * _weights.sum(["x", "y"]) <- for the global weighting
        ) ** 0.5  / misc._norm_isf_25

        da = ds.where(ds.filled_flag == 2)
        num_cells = (~da._median.isnull()).sum(["x", "y"])
        if (num_cells == 0).all():
            unc2 = xr.zeros_like(unc1)
        else:
            tmp_grouper = da.basin_id
            da = da._iqr.groupby("time")
            res = []
            print("basin")
            for label, group in tqdm.tqdm(da, desc="timesteps"):
                if tmp_grouper.sel(time=label).isnull().all():
                    continue
                grouped_group = group.groupby(tmp_grouper.sel(time=label))
                _weights = grouped_group.count()
                _ess = effective_sample_size(_weights.values)
                res.append(
                    ((grouped_group.first() * _weights) ** 2).sum("basin_id") ** 0.5 / misc._norm_isf_25 \
                    / _ess ** 0.5  # / num_cells <- for the current weighting * num_cells <- for the global weighting
                )
            unc2 = xr.concat(res, "time")

        da = ds.where(ds.filled_flag.isin([3, 4, 6]))
        num_cells = (~da._median.isnull()).sum(["x", "y"])
        if (num_cells == 0).all():
            unc3 = xr.zeros_like(unc1)
        else:
            tmp_grouper = da.group_id
            da = da._iqr.groupby("time")
            res = []
            print("group")
            for label, group in tqdm.tqdm(da, desc="timesteps"):
                if tmp_grouper.sel(time=label).isnull().all():
                    continue
                grouped_group = group.groupby(tmp_grouper.sel(time=label))
                _weights = grouped_group.count()
                _ess = effective_sample_size(_weights.values)
                res.append(
                    ((grouped_group.first() * _weights) ** 2).sum("group_id") ** 0.5 / misc._norm_isf_25 \
                    / _ess ** 0.5  # / num_cells <- for the current weighting * num_cells <- for the global weighting
                )
            unc3 = xr.concat(res, "time")

        da = ds._iqr.where(ds.filled_flag.isin([-2, 5]))
        num_cells = (~da.isnull()).sum(["x", "y"])
        unc4 = da.mean(["x", "y"]) / misc._norm_isf_25 * num_cells
        
        num_cells = (~ds.filled_flag.isnull()).sum(["x", "y"])
        _unc = xr.concat([unc1, unc2, unc3, unc4], dim="tmp") / num_cells
        results["uncertainty"] = ((_unc ** 2).sum("tmp") ** 0.5
                                * 2  # 2sigma-uncertainties
                                ).to_series()
    else:
        _coarsened = ds._iqr.coarsen(
            {"x": 4, "y": 4},
            boundary="pad",
        )
        _weights = _coarsened.count()
        results["uncertainty"] = ((
            (
                (_coarsened.mean()
                * _weights) ** 2
            ).sum(["x", "y"]) ** 0.5
            / _weights.sum(["x", "y"])
        ) ** 0.5  / misc._norm_isf_25
        * 2  # 2sigma-uncertainties
                                ).to_series()
    
    results.sort_index(axis=1, inplace=True)

    # debugging
    # print(_unc.rename("uncertainties").to_dataframe()["uncertainties"].unstack(0).to_string())
    print(results.to_string())
    import matplotlib.pyplot as plt
    plt.clf()
    plt.fill_between(results.index, results.elevation-results.uncertainty, results.elevation+results.uncertainty)
    plt.plot(results.index, results.elevation, c="k")
    plt.ylabel("Surface elevation difference, m")
    if 'o2region' not in ds.attrs:
        from pickle import dumps
        from hashlib import md5
        ds.attrs['o2region'] = md5(dumps(ds), usedforsecurity=False).hexdigest()[:7]
    plt.title(ds.attrs['o2region'])
    plt.savefig(f"tmp__quick_view_elev_ts_with_unc__{ds.attrs['o2region']}.png")

    return results


def trend_with_seasons(
    t_ns, trend, offset, amp_yearly, phase_yearly, amp_semiyr, phase_semiyr
):
    t_yr = t_ns / (365.25 * 24 * 60 * 60 * 1e9)
    return (
        offset
        + t_yr * trend
        + np.abs(amp_yearly) * np.exp((2 * np.pi * t_yr - phase_yearly) * 1j).real
        + np.abs(amp_semiyr) * np.exp((4 * np.pi * t_yr - phase_semiyr) * 1j).real
    )
