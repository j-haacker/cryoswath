import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS
import rasterio
import shapely
import warnings

from .misc import *

__all__ = list()

# ! tbi:
rgi_o1_epsg_dict = dict()


def buffer_4326_shp(shp: shapely.Geometry, radius: float, simplify: bool = True) -> shapely.MultiPolygon:
    # the algorithm splits a multi-geomerty like MultiPolygon into its
    # parts, simplifies them if requested, buffers them, and joins them
    # finally.
    # the splitting is necessary to work around issue #13
    if shp is None: # this will occasionally fail, as it will be 'GEOMETRYCOLLECTION EMPTY'
        warnings.warn("shp=None passed to buffer_4326_shp, returning empty MultiPolygon.")
        return shapely.MultiPolygon()
    # # below does not take the geopandas detour. while more straight forward,
    # # geopandas used to be more stable. however the below was improved and
    # # may be equally good now.
    # transformer = Transformer.from_crs("EPSG:4326", planar_crs)
    # shp = shapely.ops.transform(transformer.invert().transform, shapely.ops.transform(transformer.transform, shp).simplify(100).buffer(radius)).make_valid()
    if isinstance(shp, shapely.geometry.base.BaseMultipartGeometry):
        shp = list(shp.geoms)
    elif isinstance(shp, shapely.geometry.base.BaseGeometry):
        # assuming single shaply geometry as it is not a collection
        shp = [shp]
    elif isinstance(shp, gpd.GeoDataFrame):
        shp = shp.geometry.values
    elif isinstance(shp, gpd.GeoSeries):
        shp = shp.values
    elif isinstance(shp, gpd.array.GeometryArray):
        # nothing to do
        pass
    planar_crs = find_planar_crs(shp=shp[0])
    buffered_planar = []
    for poly in shp:
        buffered_planar.append(
            gpd.GeoSeries(poly, crs=4326).to_crs(planar_crs).make_valid().simplify(100).buffer(radius))
    buffered_planar = pd.concat(buffered_planar)
    if simplify:
        buffered_planar = buffered_planar.simplify(radius/3)
    buffered_planar = buffered_planar.to_crs(4326)
    return shapely.make_valid(buffered_planar.union_all(method="unary"))
__all__.append("buffer_4326_shp")


def ensure_pyproj_crs(crs: CRS) -> CRS:
    # Token function to convert (any?) CRS object to a pyproj.crs.CRS
    # For now using from_epsg as it smells safest.
    if not isinstance(crs, CRS):
        try:
            epsg = crs.to_epsg()
        except AttributeError:
            epsg = crs
        crs = CRS.from_epsg(epsg)
    return crs
__all__.append("ensure_pyproj_crs")


def esri_to_feather(file_path: str = None) -> None:
    if file_path.split(os.path.extsep)[-1].lower() == "shp":
        basename = os.path.extsep.join(file_path.split(os.path.extsep)[:-1])
    gpd.read_file(file_path).to_feather(basename+os.path.extsep+"feather")
__all__.append("esri_to_feather")


def find_planar_crs(*, shp: shapely.Geometry = None, lat: float = None, lon: float = None, region_id: str = None):
    if region_id is not None:
        shp = load_glacier_outlines(region_id)
    elif shp is None:
        shp = shapely.MultiPoint([(lon, lat) for lon, lat in zip(lon, lat)])
    shp = shp.centroid
    if shp.y > 75:
        return CRS.from_epsg(3413)
    elif shp.y < -75:
        return CRS.from_epsg(3976)
    else:
        return gpd.GeoSeries(shp, crs=4326).to_frame().estimate_utm_crs()
__all__.append("find_planar_crs")

def get_lon_origin(crs):
    # Extract Longitude of origin
    # May turn out not to be very robust.
    return ensure_pyproj_crs(crs).coordinate_operation.params[1].value
__all__.append("get_lon_origin")
    

def get_4326_to_dem_Transformer(dem_reader: rasterio.DatasetReader) -> Transformer:
    return Transformer.from_crs("EPSG:4326", ensure_pyproj_crs(dem_reader.crs))
__all__.append("get_4326_to_dem_Transformer")


def points_on_glacier(points: gpd.GeoSeries) -> pd.Index:
    o2regions = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
    o2code = o2regions[o2regions.geometry.contains(shapely.box(*points.total_bounds))]["o2region"].values[0]
    buffered_glaciered_area_polygon = load_o2region(o2code)
    import time
    print(time.time(), "building union")
    union = buffered_glaciered_area_polygon.unary_union
    print(time.time(), "building geoseries")
    buffered_glaciered_area_polygon = gpd.GeoSeries(union,
                                                     # ! the buffering below works only for the arctic
                                                     crs=buffered_glaciered_area_polygon.crs)
    print(time.time(), "buffering")
    buffered_glaciered_area_polygon = buffered_glaciered_area_polygon.to_crs(3413).buffer(30_000)
    print(time.time(), "simplifying")
    buffered_glaciered_area_polygon = buffered_glaciered_area_polygon.simplify(1000).to_crs(4326)
    return points[points.within(buffered_glaciered_area_polygon[0])].index
__all__.append("points_on_glacier")


def simplify_4326_shp(shp: shapely.Geometry, tolerance: float = None) -> shapely.Geometry:
    if tolerance is None:
        if shp.length >= 20_000: # 5 x 5 km
            tolerance = 1000
        else:
            tolerance = 300
    planar_crs = find_planar_crs(shp=shp)
    # simplify can create holes outside of the polygon. this is fixed by buffer(0) or make_valid()
    if isinstance(shp, shapely.Geometry):
        shp = gpd.GeoSeries(shp, crs=4326)
    return shp.to_crs(planar_crs).simplify(tolerance).to_crs(4326).make_valid().union_all(method="unary")
__all__.append("simplify_4326_shp")


def subdivide_region(basin_gdf: gpd.GeoDataFrame,
                     lat_bin_width_degree: float = 1,
                     lon_bin_width_degree: float = 1,
                    ) -> list[gpd.GeoDataFrame]:
    """Devides GeoDataFrame of basins into smaller GeoDataFrame based on
    their central lat/lon coords.

    Args:
        basin_gdf (gpd.GeoDataFrame, optional): Basins to subdivide.
        lat_bin_width_degree (float, optional): Requested sub-region latitude
            range in degrees. Defaults to 1.
        lon_bin_width_degree (float, optional): Requested sub-region longitude
            range in degrees.. Defaults to 1.

    Returns:
        list[gpd.GeoDataFrame]: List of region parts.
    """
    return_list = []
    # cut latitude into degree slices
    n_lat_bins = max(1, round((basin_gdf.cenlat.max()-basin_gdf.cenlat.min())/lat_bin_width_degree))
    # below, `observed=True` to grant compatibility with future pandas versions.
    for lat_label, lat_group in basin_gdf.groupby(pd.cut(basin_gdf.cenlat, bins=n_lat_bins), observed=True):
        # similarly, cut longitude
        n_lon_bins = max(1, round((lat_group.cenlon.max()-lat_group.cenlon.min())
                                                     / lon_bin_width_degree
                                                     * np.cos(np.deg2rad(lat_label.mid))))
        for _, lon_group in lat_group.groupby(pd.cut(lat_group.cenlon, bins=n_lon_bins), observed=True):
            return_list.append(lon_group)
    return return_list
__all__.append("subdivide_region")
    