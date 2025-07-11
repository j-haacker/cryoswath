FROM quay.io/jupyter/minimal-notebook:latest

RUN mkdir $HOME/project_dir
WORKDIR $HOME/project_dir
RUN git clone https://github.com/j-haacker/cryoswath.git
RUN conda install numpy GDAL distributed
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN pip install --editable "./cryoswath"
RUN cryoswath-init
WORKDIR $HOME/project_dir/scripts
