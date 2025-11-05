# Official Ubuntu 22.04 base image
FROM ubuntu:22.04

# Set the environment variables to avoid user prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Set working directory
WORKDIR /app

# Copy codebase into image
COPY ./assets /app/assets
COPY ./config /app/config
COPY ./opt /app/opt
COPY ./src /app/src
COPY ./LICENSE /app/LICENSE
COPY ./README.md /app/README.md
COPY ./requirements.txt /app/requirements.txt

# Basic dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    unzip \
    curl \
    git \
    wget \
    gfortran \
    make \
    gcc \
    g++ \
    make \
    tcsh \
    bc \
	libssl-dev \
    libblas-dev \
    liblapack-dev \
    && apt-get clean

# Download utility functions
RUN git clone https://github.com/ellis-langford/ImageUtils.git /src/utils

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN python3.10 -m pip install \
--trusted-host pypi.python.org -r /app/requirements.txt

# Install Jupyter
RUN pip install jupyterlab
# Expose JupyterLab's default port (8888)
EXPOSE 8888

# ANTsPy setup
RUN pip install antspyx
ENV ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
ENV ANTS_RANDOM_SEED=1

# Download FreeSurfer
RUN cd /usr/local \
    && wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
    && tar -xzvf freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
    && rm -rf freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz

# Initialise FreeSurfer and copy license
ENV FREESURFER_HOME=/usr/local/freesurfer
RUN printf "\n# Freesurfer\nsource $FREESURFER_HOME/SetUpFreeSurfer.sh\n" >> /root/.bashrc
COPY ./assets/fs_license.txt $FREESURFER_HOME/license.txt

# Download and install PETSc
WORKDIR /opt
RUN git clone -b release https://gitlab.com/petsc/petsc.git petsc

WORKDIR /opt/petsc
RUN ./configure \
        --prefix=/usr/local/petsc \
        --with-cc=gcc \
        --with-cxx=g++ \
        --with-fc=gfortran \
        --download-f2cblaslapack \
        --download-mpich \
        --with-fortran-bindings=1 && \
    make PETSC_DIR=/opt/petsc PETSC_ARCH=arch-linux-c-debug all && \
    make PETSC_DIR=/opt/petsc PETSC_ARCH=arch-linux-c-debug install

ENV PETSC_DIR=/usr/local/petsc

# Revert working directory
WORKDIR /

# Set the command to run JupyterLab when the container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
