# import matplotlib.figure
from matplotlib import pyplot as plt
# from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Iterable, Literal
import xarray as xr

from cryoswath import gis, misc

__all__ = []


def elevation_change(*data: xr.DataArray,
                     type: Iterable[Literal["plain",
                                            "start_at_0",
                                            "cummulative",
                                            "interleafed-cummulative"]] = ("plain",),
                     pivot_month: int = 10,
                     start_date: pd.Timestamp = pd.Timestamp("2010-10"),
                     end_date: pd.Timestamp = None,
                     plot_specs: Iterable[dict[str,Any]] = ({},),
                     ax: plt.Axes = None,
                     add_legend: bool = True,
                     despine: dict[str,Any] = dict(trim=True),
                     ) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    data = tuple(data)
    for i, da in enumerate(data):
        da = da.sel(time=slice(start_date, end_date)).mean(["x", "y"])
        if type[i] == "start_at_0":
            da = da - xr.polyval(da.time[0], da.polyfit("time", 1)).polyfit_coefficients.values
        elif type[i] == "cummulative":
            da = da.cumsum("time")
        elif type[i] == "interleafed-cummulative":
            change_1st_year = da.isel(time=da.time.dt.month==pivot_month)[:2].diff("time").values
            # the reference between the interleafed series shifts. the change of
            # n/12th months of the first year needs to be added.
            da = xr.concat([da.isel(time=da.time.dt.month==month).cumsum()
                            + (month-da.time.min().dt.month.values)%12/12*change_1st_year
                            for month in range(1, 13)], "time").sortby("time")
        # only implemented for geospatial (x,y) data
        da.plot(ax=ax, **plot_specs[i])
        # da.isel(time=da.time.dt.month==pivot_month).plot(ax=ax, marker="o", ls="")
    if add_legend:
        ax.legend([da.name for da in data])
    sns.despine(**despine)
    ax.set_xlabel("") # is always time. however, its obvious and doesn't need a label
    ax.set_ylabel("average elevation change, m")
    # tbi: maybe add option to pass title and axis label
    ax.set_title("")
    return ax
__all__.append("elevation_change")
