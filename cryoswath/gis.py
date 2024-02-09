import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS
import shapely

from .misc import *

# ! tbi:
rgi_o1_epsg_dict = dict()

def buffer_4326_shp(shp, radius: float):
    # ! currently only works for the Arctic
    to_planar = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3413))
    to_4326 = Transformer.from_crs(CRS.from_epsg(3413), CRS.from_epsg(4326))
    return shapely.ops.transform(to_4326.transform, shapely.ops.transform(to_planar.transform, shp).buffer(radius))

def ensure_pyproj_crs(crs):
    # Token function to convert (any?) CRS object to a pyproj.crs.CRS
    # For now using from_epsg as it smells safest. Inflicts many dependencies
    # (crs needs to have a code, crs object needs to have .to_epsg, ...)
    if not isinstance(crs, CRS):
        crs = CRS.from_epsg(crs.to_epsg())
    return crs


def esri_to_feather(file_path):
    if file_path[-4:].lower() == ".shp":
        file_path = file_path[:-4]
    gpd.read_file(file_path).to_feather(file_path+".feather")


def get_lon_origin(crs):
    # Extract Longitude of origin
    # May turn out not to be very robust.
    return ensure_pyproj_crs(crs).coordinate_operation.params[1].value
    

def get_4326_to_dem_Transformer(dem):
    return Transformer.from_crs("EPSG:4326", ensure_pyproj_crs(dem.crs))

def points_on_glacier(points: gpd.GeoSeries) -> pd.Index:
    o2regions = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
    o2code = o2regions[o2regions.geometry.contains(shapely.box(*points.total_bounds))]["o2region"].values[0]
    buffered_glaciered_area_polygon = load_o2region(o2code)
    buffered_glaciered_area_polygon = gpd.GeoSeries(buffered_glaciered_area_polygon.unary_union,
                                                     # ! the buffering below works only for the arctic
                                                     crs=buffered_glaciered_area_polygon.crs)\
                                                         .to_crs(3413).buffer(30000).simplify(1000).to_crs(4326)
    return points[points.within(buffered_glaciered_area_polygon[0])].index
