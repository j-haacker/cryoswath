# cryoswath

cryoswath is a python package containing processing pipelines, a tool
library, and some pre-assemble data to retrieve and study CryoSat-2 data

## state

Currently, it is in the alpha phase. `main` contains those parts that I
believe to work if used as intended and that tested to some
extent. Other branches are for development.

Check the [preliminary tutorial](https://github.com/j-haacker/cryoswath/blob/main/scripts/tutorial.ipynb)
for some inspiration how to start off.

## 🚀 features

- find all CryoSat-2 tracks passing over your region of interest
- download L1b data from ESA
- retrieve swath elevation estimates
- aggregate point data to gridded data

## dependencies

cryoswath builds on a number of other packages. See
[requirements.txt](https://github.com/j-haacker/cryoswath/blob/main/requirements.txt).

## 🐛 known issues

- ! compatibility issues with xarray v2024.3.0  
    -> downgrade to 2024.1.1
- it has only been tested for the Arctic

### missing dependencies

- the reference elevation model is not provided
- the RGI data are not automatically downloaded

## 🎯 what's coming

- pipeline building gridded change rates

Further: See the [wish-list (#enhancement in GH issues)](https://github.com/j-haacker/cryoswath/labels/enhancement).

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

## license

MIT. See [LICENSE.txt](https://github.com/j-haacker/cryoswath/blob/main/LICENSE.txt).
