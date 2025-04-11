"""Smell-check results by plotting spatial aggregates"""

__all__ = ["coverage", "l2"]

import geopandas as gpd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:  # lifts seaborn dependency
    import seaborn as sns
except ImportError:
    pass

from cryoswath import misc


def coverage(
    l3_data_or_filepath: xr.Dataset | str,
    at_time: pd.Timestamp | pd.DatetimeIndex = pd.date_range(
        "2010-10-01", "2023-10-01", freq="12MS"
    ),
) -> mpl.figure.Figure:
    if isinstance(l3_data_or_filepath, str):
        l3_data = xr.open_dataset(l3_data_or_filepath, decode_coords="all")
    else:
        l3_data = l3_data_or_filepath
    l3_data = l3_data.transpose("time", "y", "x")
    # print(l3_data)
    o2_code = misc.find_region_id(l3_data, scope="o2")
    glacier_complexes = misc.load_glacier_outlines(
        o2_code, product="complexes", union=False
    )
    crs = glacier_complexes.estimate_utm_crs()
    glacier_complexes = glacier_complexes.to_crs(crs)
    ds = l3_data.sel(time=at_time).rio.reproject(crs)
    if "sns" in globals():
        sns.ecdfplot(
            ds._iqr.where(ds._count > 3)
            .fillna(999)
            .rio.clip([glacier_complexes.union_all(method="coverage")])
            .to_numpy()
            .flatten()
        )
    else:
        ds._iqr.where(ds._count > 3).fillna(999).rio.clip(
            [glacier_complexes.union_all(method="coverage")]
        ).plot.hist(bins=100, range=(0, 50), cumulative=True, histtype="step")
    plt.xlim([0, 50])
    plt.xlabel("Inter-quartile range, m")
    plt.gca().xaxis.set_label_position("top")
    if "sns" in globals():
        sns.despine(top=False, bottom=True)
    else:
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
    plt.subplots_adjust(right=0.75, bottom=0.25)
    plt.gcf().add_axes((0.33, 0, 0.67, 0.67))
    ds._iqr.where(ds._count > 3).median("time").fillna(999).rio.clip(
        [glacier_complexes.union_all(method="coverage")]
    ).plot(vmin=5, vmax=30, cbar_kwargs={"label": "Inter-quartile range, m"})
    glacier_complexes.boundary.plot(ax=plt.gca(), color="m", lw=0.5)
    plt.title("")
    plt.gca().axis("off")
    return plt.gcf()


def l2(
    swath: gpd.GeoDataFrame,
    column: str = "height",
    pocas: gpd.GeoDataFrame = None,
    glaciers: gpd.GeoSeries | gpd.GeoDataFrame = None,
    ax: mpl.axes.Axes = None,
    cmap: mpl.colors.Colormap = mpl.cm.viridis,
) -> mpl.axes.Axes:
    if ax is None:
        ax = plt.axes()
    if glaciers is not None:
        glaciers = glaciers.to_crs(swath.crs)
        bounds = glaciers.total_bounds
        glaciers.plot(
            label="glacierized", color="xkcd:ice", edgecolor="k", linewidth=0.4, ax=ax
        )
    else:
        bounds = swath.total_bounds
    norm = mpl.colors.Normalize(
        *pd.concat([swath, pocas]).clip(bounds)[column].describe()[["min", "max"]]
    )
    if pocas is not None:
        pocas = pocas.clip(bounds)
        pocas.geometry.plot(
            label="POCA", marker="o", markersize=4, color="tab:green", ax=ax
        )
        pocas.plot(ax=ax, column=column, marker=".", markersize=2, cmap=cmap, norm=norm)
    swath.clip(bounds).plot(
        label="data",
        ax=ax,
        column=column,
        marker=".",
        markersize=1,
        cmap=cmap,
        norm=norm,
    )
    for ix_lev in swath.index.levels:
        if ix_lev.dtype.kind == "M":
            trks = misc.load_cs_ground_tracks()
            trk_ix = np.unique(
                trks.index.get_indexer(ix_lev.tz_localize(None), method="ffill")
            )
            trks.iloc[trk_ix].to_crs(swath.crs).clip(bounds).plot(
                label="track", ax=ax, ls="--", color="tab:gray"
            )
    ax.legend()
    plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Elevation, m"
    )
    return ax
