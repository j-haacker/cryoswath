FROM condaforge/miniforge3

COPY conda_requirements.txt .
COPY custom_xarray.patch .

RUN mamba create --name cryoswath --file conda_requirements.txt --yes
RUN echo "/opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate cryoswath" >> /root/.bashrc
RUN apt update && apt install patch
RUN find -name variables.py -path */cryoswath/*/xarray/coding/* -exec \
    patch {} custom_xarray.patch \;

SHELL ["conda", "run", "-n", "cryoswath", "/bin/bash", "-c"]
