# ARG BASE_IMAGE_NAME=python3:cuda-11.1

# ######################################################################
# #### Base Image
# ######################################################################
# FROM $BASE_IMAGE_NAME

# # Install some basic utilities
# RUN apt-get update && apt-get install -y \
#     curl \
#     ca-certificates \
#     sudo \
#     git \
#     bzip2 \
#     libx11-6 \
#     && rm -rf /var/lib/apt/lists/*


# ENV LANG=C.UTF-8 \
#     DEBIAN_FRONTEND=noninteractive \
#     PIP_NO_CACHE_DIR=true

# ###################################
# #### Setup Python3
# ###################################
# # # make some useful symlinks that are expected to exist
# # RUN ln -s /python/bin/python3-config /usr/local/bin/python-config && \
# # 	ln -s /python/bin/python3 /usr/local/bin/python && \
# # 	ln -s /python/bin/python3 /usr/local/bin/python3 && \
# # 	ln -s /python/bin/pip3 /usr/local/bin/pip && \
# # 	ln -s /python/bin/pip3 /usr/local/bin/pip3 && \
# # 	# install depedencies
# # 	apt-get update && \
# # 	apt-get install --assume-yes --no-install-recommends ca-certificates libexpat1 libsqlite3-0 libssl1.1 && \
# # 	apt-get purge --assume-yes --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
# # 	rm -rf /var/lib/apt/lists/*

# # # copy in Python environment
# # COPY --from=builder /python /python
# # ENV PATH $PATH:/python/bin


# ###################################
# #### Install pytorch
# ###################################
# # CUDA 11.0-specific steps
# RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# # OpenCV libs
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# ###################################
# #### Add User
# ###################################
# ARG USER_ID=1000
# ARG GROUP_ID=28

# RUN addgroup --gid $GROUP_ID user
# RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user &&\
#     echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
# USER user 
# ENV PATH $PATH:/home/user/.local/bin

# ###################################
# #### Install Requirements
# ###################################

# ENV USER user
# ENV DEBIAN_FRONTEND=noninteractive \
#     HOME=/home/${USER}

# # OpenMPI
# USER root
# RUN apt-get install -y --no-install-recommends gcc gfortran libopenmpi-dev openmpi-bin openmpi-common openmpi-doc && \
#     apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# # ------------------------------------------------------------
# # Configure OpenMPI
# # ------------------------------------------------------------

# RUN rm -fr ${HOME}/.openmpi && mkdir -p ${HOME}/.openmpi
# ADD default-mca-params.conf ${HOME}/.openmpi/mca-params.conf
# RUN chown -R ${USER}:${USER} ${HOME}/.openmpi
# USER user

# # Create a working directory
# ARG HOME=/home/user
# RUN mkdir ${HOME}/code
# WORKDIR ${HOME}/code

# COPY requirements.txt ${HOME}/code/
# RUN pip3 install --upgrade --no-cache-dir pip
# RUN pip install --no-cache-dir -r requirements.txt --user

FROM nnhieu/lossland

RUN wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz