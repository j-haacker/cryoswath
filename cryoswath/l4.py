import numpy as np
import pandas as pd
import scipy.special
import xarray as xr

from .misc import data_path, find_region_id, nanoseconds_per_year
from . import l3

__all__ = list()

# notes for future development:
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
        if CI<np.min([1.5, .5+.2*np.abs(trend)]): return 1/CI
        else: return 0
    ds["weights"] = xr.apply_ufunc(trunc_weights, ds.trend_CI95, ds.trend, vectorize=True)
    return l3.fill_voids(ds.rio.write_crs(data.rio.crs), "trend")
__all__.append("fit_trend")


def differential_change(data: xr.Dataset,
                        ) -> xr.Dataset:
    shiftby = np.argwhere((data.time==data.time[0].values+pd.DateOffset(years=1)).values)[0][0]
    differences = data._median.shift(time=shiftby).dropna("time", how="all")-data._median.where(data._count>3)
    scaling_factor = .5**.5/scipy.special.erf(.5)
    uncertainties = data._iqr/2 * scaling_factor * 2 / data._count**.5
    uncertainties = (uncertainties.shift(time=shiftby).dropna("time", how="all")**2+uncertainties.where(data._count>3)**2)**.5
    def trunc_weights(CI, trend):
        if CI<np.min([15, 5+.2*np.abs(trend)]): return 1/CI
        else: return 0
    weights = xr.apply_ufunc(trunc_weights, uncertainties, differences, vectorize=True)
    data = xr.merge([differences.rename("elev_change"), uncertainties.rename("elev_change_CI95"), weights.rename("weights")])
    return l3.fill_voids(data, "elev_change")
__all__.append("differential_change")


__all__ = sorted(__all__)
