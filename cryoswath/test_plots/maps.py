import matplotlib.figure
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import xarray as xr

from cryoswath import gis, misc

__all__ = []


def coverage(l3_data_or_filepath: xr.Dataset|str,
             at_time: pd.Timestamp|pd.DatetimeIndex = pd.date_range("2010-10-01", "2023-10-01", freq="12MS"))\
        -> matplotlib.figure.Figure:
    if isinstance(l3_data_or_filepath, str):
        l3_data = xr.open_dataset(l3_data_or_filepath, decode_coords="all")
    else:
        l3_data = l3_data_or_filepath
    l3_data = l3_data.transpose('time', 'y', 'x')
    # print(l3_data)
    o2_code = misc.find_region_id(l3_data, scope="o2")
    glacier_complexes = misc.load_glacier_outlines(o2_code, product="complexes", union=False)
    crs = glacier_complexes.estimate_utm_crs()
    glacier_complexes = glacier_complexes.to_crs(crs)
    ds = l3_data.sel(time=at_time).rio.reproject(crs)
    sns.ecdfplot(ds._iqr.where(ds._count>3).fillna(999).rio.clip([glacier_complexes.union_all(method="coverage")]).to_numpy().flatten())
    plt.xlim([0, 50])
    plt.xlabel("Inter-quartile range, m")
    plt.gca().xaxis.set_label_position("top")
    sns.despine(top=False, bottom=True)
    plt.subplots_adjust(right=.75, bottom=.25)
    plt.gcf().add_axes((.33, 0, .67, .67))
    ds._iqr.where(ds._count>3).median("time").fillna(999).rio.clip([glacier_complexes.union_all(method="coverage")]).plot(vmin=5, vmax=30, cbar_kwargs={"label": "Inter-quartile range, m"})
    glacier_complexes.boundary.plot(ax=plt.gca(), color="m", lw=.5)
    plt.title("")
    plt.gca().axis("off")
    return plt.gcf()
__all__.append("coverage")
