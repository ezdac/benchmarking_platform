FROM rgrubba/debian-squeeze

RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install --force-yes -y \
 flex\
 bison\
 build-essential\
 python-numpy\
 cmake\
 python-dev\
 sqlite3\
 libsqlite3-dev\
 libboost-dev\
 libboost-python-dev\
 libboost-regex-dev\
# swig2.0\
 wget\
 zip
 git\

# make sure to use all the old legacy software available in Feb 2013;
# e.g. CMake 3 is not the same API as used by this version, etc. etc.
ENV RDKIT_BRANCH=Release_2013_03_2

# TODO switch to old git version, where --single-branch was not present
#   the rdkit repo seems to be very big as a whole!
#RUN git clone -b $RDKIT_BRANCH --single-branch https://github.com/rdkit/rdkit.git

ENV RDBASE=/rdkit
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RDBASE/lib
ENV PYTHONPATH=$PYTHONPATH:$RDBASE

# Avalon support
RUN mkdir /avalon
ADD AvalonToolkit_1.0.source /avalon/
ENV AVALONTOOLS_DIR=/avalon

# InChi support
WORKDIR $RDBASE/External/INCHI-API/
RUN mkdir src
ADD inchi-api-source src/

RUN mkdir $RDBASE/build
WORKDIR $RDBASE/build
#RUN cmake --trace --debug-output -Wdev -DRDK_BUILD_INCHI_SUPPORT=ON -DRDK_BUILD_AVALON_SUPPORT=ON ..
#RUN make
#RUN make install

#
## install dependencies for the benchmarking scripts
#RUN apt-get install -y \
#python-pip \
#python-setuptools
#
#RUN apt-get install -y \
#libatlas-base-dev \
#gcc \
#gfortran \
#g++
#
#WORKDIR /
#RUN pip install scipy==0.12.0
#RUN pip install scikit-learn==0.13
#
#ADD . /benchmark
#
#WORKDIR /benchmark