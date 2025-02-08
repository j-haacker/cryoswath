# cryoswath

![GitHub top language](https://img.shields.io/github/languages/top/j-haacker/cryoswath)
![PyPI - Version](https://img.shields.io/pypi/v/cryoswath)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14837018.svg)](https://doi.org/10.5281/zenodo.14837018)
![GitHub License](https://img.shields.io/github/license/j-haacker/cryoswath)

cryoswath is a python package containing processing pipelines, a tool
library, and some pre-assembled data to retrieve and study CryoSat-2
data.

Adaptability lies at its core. The user can access many options simply
by passing arguments to functions; everything else can be customized
changing the concerned function or adding a new one.

## 🌱 state

cryoswath is being developed. Branch `main` contains the
"pip"-installable package, `scripts` contains tutorials, and `data`
contains auxiliary data and the required directory structure. You can
have everything setup automatically (see "getting started"). Other
branches are for development.

## ✨ features

- find all CryoSat-2 tracks passing over your region of interest
- download L1b data from ESA
- retrieve swath elevation estimates
- aggregate point data to gridded data
- fill data gaps using tested methods
- calculate change rates

## 🚀 getting started

There is a number of ways you can start off. I will give detailed
instructions for UNIX systems. Make sure to use python 3.11 or higher.
Further, I recommend to use a virtual environment and will involve
python-venv in the instructions (however, conda works similar).

### with git 🐙

advantage: easy pulling bugfixes

Set up a project directory, pull this repo, create virtual
environment, initialize, and download ArcticDEM and the RGI glacier and complex
shape files into the `data/auxiliary/DEM` and -`RGI` directories.

```sh
proj_dir=altimetry-project
git clone https://github.com/j-haacker/cryoswath.git $proj_dir
cd $proj_dir
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cryoswath-init
```

### with pip 📦

advantage: easy installation

Set up a project directory, create virtual environment, install
cryoswath, initialize, and download ArcticDEM and the RGI glacier and
complex shape files into the `data/auxiliary/DEM` and -`RGI`
directories.

```sh
proj_dir=altimetry-project
cd $proj_dir
python3.11 -m venv .venv
source .venv/bin/activate
pip install cryoswath
cryoswath-init
```

### Docker and conda

New setup instructions coming soon.

### multiple projects

Similar to the above, set up a virtual environment but rather locate it
in a neutral directory. For each project, run `cryoswath-init`.

## 📖 documentation

[j-haacker.github.io/cryoswath](https://j-haacker.github.io/cryoswath/)

## dependencies

- [requirements.txt](https://github.com/j-haacker/cryoswath/blob/main/requirements.txt)
- reference elevation model
- glacier outlines

cryoswath will point you to the required resources.

## 🐛 known issues

- projected RGI basins sometimes "invalid"
    -> add `.make_valid()` if it is missing somewhere
- it has mostly been tested for the Arctic

  Further: see [open issues](https://github.com/j-haacker/cryoswath/issues).

## citation and attribution

You can cite this package using bibtex:

```bibtex
@software{cryoswath,
  author       = {Haacker, Jan},
  title        = {cryoswath: v0.2.1},
  month        = feb,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v0.2.1},
  doi          = {10.5281/zenodo.14837018}
}
```

Please mind that you likely used other resources on the way.

- ESA provides the L1b data under [these Terms and Conditions](https://github.com/j-haacker/cryoswath/blob/main/data/L1b/Terms-and-Conditions-for-the-use-of-ESA-Data.pdf)
- RGI data is distributed under [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/)
- if you (likely) used DEMs of the PGC, see their [Acknowledgement Policy](https://www.pgc.umn.edu/guides/user-services/acknowledgement-policy/)
- the many python packages and libraries this package depends on; some of which are indispensable.

## 📜 license

MIT. See [LICENSE.txt](https://github.com/j-haacker/cryoswath/blob/main/LICENSE.txt).
