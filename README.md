# cryoswath

![GitHub top language](https://img.shields.io/github/languages/top/j-haacker/cryoswath)
![Conda Version](https://img.shields.io/conda/vn/conda-forge/cryoswath)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14825358.svg)](https://doi.org/10.5281/zenodo.14825358)
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

All of the following instructions consist of three main steps:

1. making cryoswath available
2. setting up a project directory
3. initializing cryoswath

The instructions will use the variable `$proj_dir` for the project
directory. Please set it to a path that suits you like
`proj_dir=altimetry-project`.

In all cases, consider to download the data dependencies ArcticDEM and
the RGI glacier and complex shape files into the
`$proj_dir/data/auxiliary/DEM` and `$proj_dir/data/auxiliary/RGI`
directories (more in the [docs](https://j-haacker.github.io/cryoswath/prerequisites.html)).

### with git 🐙

advantage: easy pulling bugfixes

```sh
git clone https://github.com/j-haacker/cryoswath.git $proj_dir
cd $proj_dir
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cryoswath-init
```

### with pip 📦

advantage: easy installation

```sh
mkdir $proj_dir
cd $proj_dir
python3.11 -m venv .venv
source .venv/bin/activate
pip install cryoswath
cryoswath-init
```

### with conda 🐍

advantage: most stable dependency resolution

First, choose an environment name and either define `$env_name`, e.g.,
`env_name=cryoswath`, or adapt the create and activate commands
accordingly.

```sh
conda create -n $env_name conda-forge::cryoswath
conda activate $env_name
mkdir $proj_dir
cd $proj_dir
cryoswath-init
```

### with Docker 🐳

advantage: will almost always work

*note*: the first time running the docker image require to download ~ 1 Gb

1. `docker run -it -p 8888:8888 -v $proj_dir$:/home/jovyan cryoswath/jupyterlab:v0.2.1`
2. You will receive an address including a token with which you can connect to the jupyterlab using your browser
3. In jupyterlab, open a regular shell and execute `cryoswath-init`
4. Open the scripts folder in the explorer and select one of the notebooks or create your own (inside the scripts folder)

### multiple projects

For each project, run `cryoswath-init` from the project directory.

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
