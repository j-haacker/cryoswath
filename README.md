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

cryoswath is being developed. Branch `main` is the release branch,
`scripts` contains tutorials, and `data` contains auxiliary data and the
required directory structure. You can have everything setup
automatically (see "getting started"). Other branches are for
development.

## ✨ features

- find all CryoSat-2 tracks passing over your region of interest
- download L1b data from ESA
- retrieve swath elevation estimates
- aggregate point data to gridded data
- fill data gaps using tested methods
- calculate change rates

## 🚀 getting started

There is a number of ways you can start off, including installing from
"source", pypi, conda-forge, or docker. Please find more details in the
[docs
(prerequisites)](https://cryoswath.readthedocs.io/en/latest/prerequisites.html).
I show two approaches, installing from conda-forge and a mixture of methods.

### simply with mamba/conda 🐍

advantage: simple and most stable dependency resolution

First, choose an environment name and either define `$env_name`, e.g.,
`env_name=cryoswath`, or adapt the create and activate commands
accordingly.

`mamba create -n $env_name conda-forge::cryoswath`

Continue below at "init project".

### clone 🐙, mamba 🐍, pip 📦

advantage: allows modifications and easy updates

Like the above, first, choose an environment name and either define
`$env_name`, e.g., `env_name=cryoswath`, or adapt the create and
activate commands accordingly. You will also need the path to your
environment. That will be something ending in `.../envs/env_name`. If
you are not sure, find it viewing `mamba env list`. Further, I assume
you'll clone into a directory named `cryoswath`.

```sh
git clone https://github.com/j-haacker/cryoswath.git cryoswath
mamba env create -n $env_name -f cryoswath/environment.yml
mamba activate $env_name
mamba install pip
pip install --editable cryoswath
```

### init project

cryoswath will deal with data that is not meant to reside in the
installation directory. The command `cryoswath-init` will setup a
directory structure and download some auxiliary files. Please choose a
project name of you liking and replace `proj_dir` in the following.

```sh
mkdir proj_dir
cd proj_dir
cryoswath-init
```

This, among others, creates a file `scripts/config.ini` that contains
the base path of your project. This allow cryoswath to find the data -
if you wish to run scripts from different directories, copy this file
there.

### if everything fails: Docker 🐳

advantage: will almost always work

*note*: the first time running the docker image requires to download ~ 1 Gb

1. `docker run -it -p 8888:8888 -v <proj_dir>:/home/jovyan/project_dir cryoswath/jupyterlab:v0.2.3`
2. You will receive an address including a token with which you can connect to the jupyterlab using your browser
3. Start doing your things or get started with the installed tutorial notebooks!

## 📖 documentation

[cryoswath.readthedocs.io](https://cryoswath.readthedocs.io/)

## dependencies

- [requirements.txt](https://github.com/j-haacker/cryoswath/blob/main/requirements.txt)
- reference elevation model
- glacier outlines

cryoswath will point you to the required resources.

## 🐛 known issues

- ESA's data server is not available from all internet service providers
- projected RGI basins sometimes "invalid"
    -> add `.make_valid()` if it is missing somewhere
- it has mostly been tested for the Arctic

  Further: see [open issues](https://github.com/j-haacker/cryoswath/issues).

## citation and attribution

You can cite this package using bibtex:

```bibtex
@software{cryoswath,
  author       = {Haacker, Jan},
  title        = {cryoswath: v0.2.3},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v0.2.3},
  doi          = {10.5281/zenodo.15809596}
}
```

Please mind that you likely used other resources on the way.

- ESA provides the L1b data under [these Terms and Conditions](https://github.com/j-haacker/cryoswath/blob/main/data/L1b/Terms-and-Conditions-for-the-use-of-ESA-Data.pdf)
- RGI data is distributed under [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/)
- if you (likely) used DEMs of the PGC, see their [Acknowledgement Policy](https://www.pgc.umn.edu/guides/user-services/acknowledgement-policy/)
- the many python packages and libraries this package depends on; some of which are indispensable.

## 📜 license

MIT. See [LICENSE.txt](https://github.com/j-haacker/cryoswath/blob/main/LICENSE.txt).
