Prerequisites
=============

.. _install:

Installation
------------

To install cryoswath, clone the GitHub repository.

``git clone git@github.com/j-haacker/cryoswath.git``

Then install it into your projects virtual environment. If you don't use
virtual environments, please do so for this project to ensure all
dependencies can be satisfied. You can install cryoswath by running:

``pip install --editable ./cryoswath``

Supplying the ``--editable`` flag allows you to update and modify cryoswath.

Continue by running ``cryoswath-init``.

This will setup a directory structure, download the package, and download some small auxiliary files.
Large resource dependency need to be downloaded manually.

The above installation is the recommended. Refer to the readme for a
variety of alternatives that may help incase of issues. Among the
alternatives, there is a fallback strategy using a docker container.

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

The software dependencies should have been installed when you ran ``pip
install --editable ./cryoswath`` (see above). The dependencies can be
found in the ``pyproject.toml``. There is a historic `requirements.txt
<https://github.com/j-haacker/cryoswath/blob/main/requirements.txt>`_
that used to list dependencies that are needed or beneficial to run
cryoswath. However, it is not well maintained.
Note, that the package names are "conda" names; "pip" names my be
slightly different.
