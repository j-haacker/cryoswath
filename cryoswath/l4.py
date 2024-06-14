import numpy as np
import pandas as pd
import xarray as xr

from . import l3

__all__ = list()


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
    fit_res = data.sel(time=time_indices).polyfit("time", 1, cov=True)
    nanoseconds_per_year = 365.25*24*60*60*1e9
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
        if CI<np.min([1.5, .5+.2*np.abs(trend)]): return 1/CI
        else: return 0
    ds["weights"] = xr.apply_ufunc(trunc_weights, ds.trend_CI95, ds.trend, vectorize=True)
    return l3.fill_voids(ds.rio.write_crs(data.rio.crs), "trend")
__all__.append("fit_trend")


__all__ = sorted(__all__)
