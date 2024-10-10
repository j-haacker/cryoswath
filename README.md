# cryoswath

![GitHub Tag](https://img.shields.io/github/v/tag/j-haacker/cryoswath)
![GitHub top language](https://img.shields.io/github/languages/top/j-haacker/cryoswath)
![GitHub License](https://img.shields.io/github/license/j-haacker/cryoswath)


cryoswath is a python package containing processing pipelines, a tool
library, and some pre-assemble data to retrieve and study CryoSat-2 data

## 🌱 state

Currently, it is in the beta phase. `main` contains those parts that I
believe to work if used as intended and that tested to some
extent. Other branches are for development.

## ✨ features

- find all CryoSat-2 tracks passing over your region of interest
- download L1b data from ESA
- retrieve swath elevation estimates
- aggregate point data to gridded data
- fill data gaps using tested methods
- calculate change rates

## 🚀 getting started

To use cryoswath, pull this repo and download ArcticDEM and the RGI
glacier and complex shape files into the `data/auxiliary/DEM` and -`RGI`
directories. Then, either use the provided docker container or set up an
environment and install the software dependencies. If you choose the
latter, you will have to either limit xarray to <2024.3 or to apply a
small patch.

### with Docker 🐳

1. `docker run --detach --interactive --volume <base dir>:/altimetry_project cryoswath/cryoswath:nightly`
2. connect with your favorite IDE or `docker exec --interactive <container hash> sh`

### with conda 🐍

1. `conda create --name env_name --file <base dir>/docker/conda_requirements.txt`
2. `conda activate env_name`
3. `conda install patch`
4. `find -name variables.py -path */env_name/*/xarray/coding/* -exec patch {} <base dir>/docker/custom_xarray.patch \;` (the patch works for `xarray=2024.9.0`)

(Steps 4+5 are necessary to change `*=` to `x=x*` in the xarray code.)

## 📖 documentation

[j-haacker.github.io/cryoswath](https://j-haacker.github.io/cryoswath/)

## dependencies

- [requirements.txt](https://github.com/j-haacker/cryoswath/blob/main/requirements.txt)
- reference elevation model
- glacier outlines

cryoswath will point you to the required resources.

## 🐛 known issues

- ! compatibility issues with xarray >= v2024.3.0  
    -> apply patch as described in getting started -> conda -> steps 4+5  
    -> \or downgrade to 2024.1.1  
    If not fixed, xarray throws an error when you load the raw data. The
    error message points you to a line in the function
    `_scale_offset_decoding` where `x*=y` needs to be spelled out as
    `x=x*y` because the operator is not defined for the data type.
- projected RGI basins sometimes "invalid"
    -> add `.make_valid()` if it is missing somewhere
- it has mostly been tested for the Arctic

  Further: see [open issues](https://github.com/j-haacker/cryoswath/issues).

## citation and attribution

You can cite this package using bibtex:

```bibtex
@misc{cryoswath,
  author = {J. Haacker},
  title = {cryoswath: CryoSat-2 swath processing package},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/j-haacker/cryoswath}}
}
```

Please mind that you likely used other resources on the way.

- ESA provides the L1b data under [these Terms and Conditions](https://github.com/j-haacker/cryoswath/blob/main/data/L1b/Terms-and-Conditions-for-the-use-of-ESA-Data.pdf)
- RGI data is distributed under [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/)
- if you (likely) used DEMs of the PGC, see their [Acknowledgement Policy](https://www.pgc.umn.edu/guides/user-services/acknowledgement-policy/)
- the many python packages and libraries this package depends on; some of which are indispensable.

## 📜 license

MIT. See [LICENSE.txt](https://github.com/j-haacker/cryoswath/blob/main/LICENSE.txt).
