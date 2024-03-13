import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS
import rasterio
import shapely

from .misc import *

__all__ = list()

# ! tbi:
rgi_o1_epsg_dict = dict()


def buffer_4326_shp(shp: shapely.Geometry, radius: float, simplify: bool = True) -> shapely.MultiPolygon:
    # ! currently only works for the Arctic
    buffered_planar = gpd.GeoSeries(shp, crs=4326).to_crs(3413).buffer(radius)
    if simplify: return buffered_planar.simplify(radius/2).to_crs(4326).iloc[0]
    return buffered_planar.to_crs(4326).iloc[0]
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
    # ! currently only works for the Arctic
    if tolerance is None:
        if shp.length >= 20_000: # 5 x 5 km
            tolerance = 1000
        else:
            tolerance = 300
    return gpd.GeoSeries(shp, crs=4326).to_crs(3413).simplify(tolerance).to_crs(4326).iloc[0]
__all__.append("simplify_4326_shp")
    