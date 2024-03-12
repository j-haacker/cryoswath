import pandas as pd
import xarray as xr

__all__ = list()


def from_l3(l3_data: xr.Dataset, *,
            pivot: pd.DateOffset, # ? best class?
            timestep_months: int = 12,
            ) -> xr.Dataset:
    # ! check l3 has no/little voids
    time_indices = pd.date_range(l3_data.time[0].values+pivot, l3_data.time[-1].values, freq=f"{timestep_months}MS")
    # ! diff only the elevation variable
    l4_data = l3_data.sel(time=time_indices).diff("time")
    # ! calculate uncertainties
    # ! build path; make optional
    # l4_data.to_netcdf()
    return l4_data
__all__.append("from_l3")


__all__ = sorted(__all__)
