Prerequisites
=============

.. _install:

Installation
------------

*note: if you wish to install the optional pdemtools, see below before
proceeding*

To install cryoswath, clone the GitHub repository.

``git clone git@github.com/j-haacker/cryoswath.git``

Then install it into your projects virtual environment. If you don't use
virtual environments, please do so for this project to ensure all
dependencies can be satisfied. You can install CryoSwath by running:

``pip install --editable <path that you cloned into>``

Supplying the ``--editable`` flag allows you to update and modify
CryoSwath.

Continue by running ``cryoswath-init``.

This will setup a directory structure, download the package, and
download some small auxiliary files. Large resource dependency need to
be downloaded manually.

The above installation is the recommended. Refer to the readme for a
variety of alternatives that may help incase of issues. Among the
alternatives, there is a fallback strategy using a docker container.

Data dependencies
-----------------

CryoSwath needs a reference elevation model. Currently, ArcticDEM and
REMA of the Polar Geospatial Center, University of Minnesota
(https://www.pgc.umn.edu/data/) are supported. To use other sources, add
their paths to :func:`cryoswath.misc.get_dem_reader`. Deposit them in
``data/auxiliary/DEM`` or change ``dem_path`` in :mod:`cryoswath.misc`
to your needs.

Further, if you would like to take advantage of the basin shapes
provided in the Randolph Glacier Inventory, download them as needed.
Make sure to download both products, "G" (glaciers/basins) and "C"
(complexes). CryoSwath will give you hints, if any data is missing, as
you go. Deposit the shape files in ``data/auxiliary/RGI`` or change
``rgi_path`` in :mod:`cryoswath.misc` to your needs. If you already know
what data you need, find them at the `nsidc repository
<https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/>`_.

Software dependencies
---------------------

The software dependencies should have been installed when you ran ``pip
install --editable ./cryoswath`` (see above). The dependencies can be
found in the ``pyproject.toml``. There is a historic `requirements.txt
<https://github.com/j-haacker/cryoswath/blob/main/requirements.txt>`_
that used to list dependencies that are needed or beneficial to run
CryoSwath. However, it is not well maintained.
Note, that the package names are "conda" names; "pip" names my be
slightly different.

Optional pdemtools
------------------

Optionally, you can install pdemtools to download data of the PGC DEMs
ArcticDEM or REMA on the fly. This can be handy if you explore small
regions. However, the data will be downloaded at a 32 m resolution -
much higher than needed in the context of CryoSat observations.

Due to pdemtools' dependence on GDAL, the installation is difficult. The
first method is still comparably simple, but will not allow to modify
CryoSwath once it is installed. The second method is issue- and
failure-prone because of various version requriements. The third option
will likely install but could lead to errors down the road.

First - conda-only:
~~~~~~~~~~~~~~~~~~~

Create a fresh conda environment installing CryoSwath and pdemtools:
:code:`conda create -n new_env cryoswath pdemtools`
(from v0.2.3 this will not be necessary anymore)

Second - system(Linux) + pip:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First install "libgdal-dev", then :code:`pip install
"gdal==$(gdal-config --version)"`, and finalize with :code:`pip install
"cryoswath[dem]"`. On other operating systems this will be similar:
importantly, make sure the libgdal is available before the installation.

Third - conda + pip:
~~~~~~~~~~~~~~~~~~~~

First create a conda environment with pdemtools and then pip-install
CryoSwath (either editable like described above or from pypi).

.. code-block:: bash

    conda create -n new_env pdemtools
    conda activate new_env
    git clone git@github.com/j-haacker/cryoswath.git cryoswath
    pip install --editable "cryoswath[dem]"
