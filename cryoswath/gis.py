import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS
import rasterio
import shapely

from .misc import *

# ! tbi:
rgi_o1_epsg_dict = dict()

def buffer_4326_shp(shp, radius: float, simplify: bool = True):
    # ! currently only works for the Arctic
    to_planar = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3413))
    to_4326 = Transformer.from_crs(CRS.from_epsg(3413), CRS.from_epsg(4326))
    buffered = shapely.ops.transform(to_planar.transform, shp).buffer(radius)
    if simplify: buffered = buffered.simplify(radius/2)
    return shapely.ops.transform(to_4326.transform, buffered)


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


def esri_to_feather(file_path):
    if file_path[-4:].lower() == ".shp":
        file_path = file_path[:-4]
    gpd.read_file(file_path).to_feather(file_path+".feather")


def get_lon_origin(crs):
    # Extract Longitude of origin
    # May turn out not to be very robust.
    return ensure_pyproj_crs(crs).coordinate_operation.params[1].value
    

def get_4326_to_dem_Transformer(dem_reader: rasterio.DatasetReader) -> Transformer:
    return Transformer.from_crs("EPSG:4326", ensure_pyproj_crs(dem_reader.crs))


def points_on_glacier(points: gpd.GeoSeries) -> pd.Index:
    o2regions = gpd.read_feather(os.path.join(rgi_path, "RGI2000-v7.0-o2regions.feather"))
    o2code = o2regions[o2regions.geometry.contains(shapely.box(*points.total_bounds))]["o2region"].values[0]
    buffered_glaciered_area_polygon = load_o2region(o2code)
    buffered_glaciered_area_polygon = gpd.GeoSeries(buffered_glaciered_area_polygon.unary_union,
                                                     # ! the buffering below works only for the arctic
                                                     crs=buffered_glaciered_area_polygon.crs)\
                                                         .to_crs(3413).buffer(30000).simplify(1000).to_crs(4326)
    return points[points.within(buffered_glaciered_area_polygon[0])].index
