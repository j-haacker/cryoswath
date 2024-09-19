from datetime import datetime
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import scipy.special
import xarray as xr

from .misc import l4_path, find_region_id, load_glacier_outlines, nanoseconds_per_year
from . import l3

__all__ = list()

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


def fit_trend(data: xr.Dataset, *,
            pivot: pd.DateOffset, # ? best class?
            timestep_months: int = 12,
            return_raw: bool = False,
            ) -> xr.Dataset:
    # using resample(time="...").nearest(pd.Timedelta(..., "days")).dropna("time", "all")
    # it could theoretically be implemented to select a valid value in the
    # proximity of the desired time stamp. because the required frequency is
    # difficult to define flexibly and for the benefit of a well-defined
    # time stamp, a different approach is taken. For a longer, e.g.,
    # 3-monthly, aggregation time, the current approach should work equally
    # fine.
    time_indices = pd.date_range(data.time[0].values+pivot, data.time[-1].values, freq=f"{timestep_months}MS")
    data = data.sel(time=time_indices)
    data = data.where(data.isel(time=slice(None,3)).any("time")).where(data.isel(time=slice(-3,None)).any("time"))
    fit_res = data.polyfit("time", 1, cov=True)
    fit_res["polyfit_coefficients"][0] = fit_res["polyfit_coefficients"][0] * nanoseconds_per_year
    fit_res["polyfit_covariance"][0,0] = fit_res["polyfit_covariance"][0,0] * nanoseconds_per_year**2
    fit_res["polyfit_covariance"][0,1] = fit_res["polyfit_covariance"][0,1] * nanoseconds_per_year
    fit_res["polyfit_covariance"][1,0] = fit_res["polyfit_covariance"][1,0] * nanoseconds_per_year
    if return_raw:
        return fit_res
    ds = xr.Dataset()
    ds["trend"] = fit_res.polyfit_coefficients.sel(degree=1)
    ds["trend_CI95"] = 2*fit_res.polyfit_covariance.isel(cov_i=0, cov_j=0)**.5
    def trunc_weights(CI, trend):
        if CI<np.min([1.5, .5+.2*np.abs(trend)]): return 2/CI
        else: return 0
    ds["weights"] = xr.apply_ufunc(trunc_weights, ds.trend_CI95, ds.trend, vectorize=True)
    return l3.fill_voids(ds.rio.write_crs(data.rio.crs), "trend", "trend_CI95")
__all__.append("fit_trend")


def difference_to_reference_dem(l3_data: xr.Dataset,
                                save_to_disk: str|bool = True,
                                basin_shapes: gpd.GeoDataFrame = None,
                                ) -> xr.Dataset:
    # roughly filter data. important: _iqr=nan will be filled   
    l3_data = l3_data.where(l3_data._count>3).where(np.abs(l3_data._median)<150)
    res = l3.fill_voids(l3_data,
                        main_var="_median", error="_iqr", elev="ref_elev",
                        basin_shapes=basin_shapes,
                        per=("basin", "basin_group"), outlier_limit=2, outlier_replace=True, outlier_iterations=3) # 
    if save_to_disk:
        try:
            region_id = find_region_id(l3_data)
        except Exception as err:
            import traceback
            print(traceback.format_exc())
            print(str(err))
            region_id = str(datetime.now())
        res.drop_encoding().to_netcdf(os.path.join(l4_path, save_to_disk if isinstance(save_to_disk, str)
                                   else region_id+"__elev_diff_to_ref_at_monthly_intervals.nc"))
    return res
__all__.append("difference_to_reference_dem")


def differential_change(data: xr.Dataset,
                        save_to_disk: str|bool = True,
                        ) -> xr.Dataset:
    # ! needs to be tested again

    # roughly filter data
    data = data.where(data._count>3).where(np.abs(data._median)<150)
    # the `shift` below sets back the data 1 year, such that this translates
    # to "later minus earlier"
    shiftby = np.argwhere((data.time==data.time[0].values+pd.DateOffset(years=1)).values)[0][0]
    differences = (data._median - data._median.shift(time=shiftby)).dropna("time", how="all")
    uncertainties = (data._iqr**2 + data._iqr.shift(time=shiftby)**2).dropna("time", how="all")**.5
    if "ref_elev" not in data:
        data = l3.append_elevation_reference(data, ref_elev_name="ref_elev")
    data = xr.merge([differences.rename("elev_change"), uncertainties.rename("elev_change_CI95"), data.ref_elev])
    res = l3.fill_voids(data, "elev_change", error="elev_change_CI95", elev="ref_elev", per=("basin", "basin_group"), outlier_limit=2, outlier_replace=True, outlier_iterations=3)
    if save_to_disk:
        res.to_netcdf(os.path.join(l4_path, save_to_disk if isinstance(save_to_disk, str) else find_region_id(data)+"__yearly_changes_at_monthly_intervals.nc"))
    return res
__all__.append("differential_change")


def relative_change(l3_data: xr.Dataset,
                    basin_shapes: gpd.GeoDataFrame = None,
                    glac_ref_year: int = 2010,
                    pivot_month: int = 9,
                    save_to_disk: str|bool = True,
                    ) -> xr.Dataset:
    # ! needs to be tested

    if isinstance(basin_shapes, str):
        basin_shapes = gpd.GeoSeries(load_glacier_outlines(basin_shapes, "glaciers", False))
    if glac_ref_year < 2010:
        glac_ref_year += 2000
    # roughly filter data
    l3_data = l3_data.where(l3_data._count>3).where(np.abs(l3_data._median)<150)
    ref_period = pd.date_range(f"{glac_ref_year}-{pivot_month:02d}", freq="MS", periods=12)
    reference = xr.Dataset(dict(_median=(("month", "x", "y"), l3_data._median.sel(time=ref_period).values),
                                _iqr=(("month", "x", "y"), l3_data._iqr.sel(time=ref_period).values)),
                                coords={"month": ref_period.month, "x": l3_data.x, "y": l3_data.y})
    values = (l3_data._median - reference._median.sel(month=l3_data.time.dt.month)
              ).drop("month")
    uncertainties = (l3_data._iqr**2 + reference._iqr.sel(month=l3_data.time.dt.month)**2
                     ).drop("month")**.5
    if "ref_elev" not in l3_data:
        l3_data = l3.append_elevation_reference(l3_data, ref_elev_name="ref_elev")
    l3_data = xr.merge([values.rename("elevation"), uncertainties.rename("error"), l3_data.ref_elev])
    res = l3.fill_voids(l3_data.drop_sel(time=ref_period),
                        main_var="elevation", error="error", elev="ref_elev",
                        basin_shapes=basin_shapes,
                        per=("basin", "basin_group"), outlier_limit=2, outlier_replace=True, outlier_iterations=3)
    l3_data["error"] = xr.where(l3_data.time.isin(ref_period), 0, l3_data.error)
    res = xr.merge([res, l3_data.sel(time=ref_period).fillna(0)], join="outer", compat="override")
    if save_to_disk:
        res.to_netcdf(os.path.join(l4_path, save_to_disk if isinstance(save_to_disk, str) else find_region_id(l3_data)+"__relative_elevation_estimates_at_monthly_intervals.nc"))
    return res
__all__.append("relative_change")


__all__ = sorted(__all__)
