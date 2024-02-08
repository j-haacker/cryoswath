# cryoswath

cryoswath is a python package containing processing pipelines, a tool library, and some pre-assemble data to retrieve and study CryoSat-2 data

## State

Currently, it is in the pre-beta phase. main contains those parts that I believe to work if used as intended and that have been tested to some extent. devel contains parts that I used successfully - use at your own risk.

## 🚀 Features

- find all CryoSat-2 tracks passing over your region of interest
- download L1b data from ESA
- retrieve swath elevation estimates

## ⬆️ Dependencies

cryoswath builds on a number of other packages. See [requirements.txt](https://github.com/j-haacker/cryoswath/main/requirements.txt).

## 🐛 Known issues

### Missing dependencies

- the RGI data are not automatically downloaded

### Limitations

- only working for the Arctic at the moment (hard coded CRS)

### other issues

- paths (partly) only for UNIX systems

## 🎯 What's coming

- pipeline building gridded change rates

Further: See [wish-list.md](https://github.com/j-haacker/cryoswath/main/wish-list.md).

## 📃 Citation

```bibtex
@misc{cryoswath,
  author = {J. Haacker},
  title = {cryoswath: CryoSat-2 processing package},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/j-haacker/cryoswath}}
}
```

## License

MIT. See [LICENSE.txt](https://github.com/j-haacker/cryoswath/main/LICENSE.txt).
