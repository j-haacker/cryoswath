Prerequisites
=============

.. _install:

Installation
------------

To install cryoswath, simply clone the GitHub repository.

``git clone git@github.com/j-haacker/cryoswath.git``

This will setup a directory structure, download the package, and download some small auxiliary files.
Large resource dependency need to be downloaded manually.

Data dependencies
-----------------

cryoswath needs a reference elevation model.
Currently, ArcticDEM and REMA of the Polar Geospatial Center, University of Minnesota (https://www.pgc.umn.edu/data/) are supported.
To use other sources, add their paths to :func:`cryoswath.misc.get_dem_reader`, e.g., `lines following 459 (frozen, maybe different from current version) <https://github.com/j-haacker/cryoswath/blob/ed0115618c9f695aa647eb2fe5a4efb61f6050e3/cryoswath/misc.py#L459>`_.
Deposit them in ``data/auxiliary/DEM`` or change ``dem_path`` in :mod:`cryoswath.misc` to your needs.

Further, if you would like to take advantage of the basin shapes provided in the Randolph Glacier Inventory, download them as needed.
Make sure to download both products, "G" (glaciers/basins) and "C" (complexes).
cryoswath will give you hints, if any data is missing, as you go.
Deposit the shape files in ``data/auxiliary/RGI`` or change ``rgi_path`` in :mod:`cryoswath.misc` to your needs.
If you already know what data you need, find them at the `nsidc repository
<https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/>`_.

Software dependencies
---------------------

There is a bunge of packages, listed in the `requirements.txt <https://github.com/j-haacker/cryoswath/blob/main/requirements.txt>`_, that are needed or beneficial to run cryoswath.
Note, that the package names are "conda" names; "pip" names my be slightly different.
Unfortunately there is an issue with ESA's L1b data: when those are read, some values are scaled.
However, the operation used by "xarray" requires the scaling factor to be of a different type.
The two easiest work-arounds are to either patch xarray, or to restrict xarrays version to "<2024.3".

I provide a docker container, including the patched xarray version.
To fire-up docker, run:

``docker run --detach --interactive --volume <base dir>:/altimetry_project cryoswath/cryoswath:nightly``

Then, connect with your favorite IDE or ``docker exec --interactive <container hash> sh``.

For the longer term, you may want to have your own environment. If you using conda, follow the steps below:

1. ``conda create --name env_name --file <base dir>/docker/conda_requirements.txt``
2. ``conda activate env_name``
3. ``conda install patch``
4. ``find -name variables.py -path */env_name/*/xarray/coding/* -exec patch {} <base dir>/docker/custom_xarray.patch \;`` (the patch works for ``xarray=2024.9.0`` which listed in the requirements.txt used above)
