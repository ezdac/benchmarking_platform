#FROM rgrubba/debian-squeeze
FROM debian:jessie

RUN apt-get update && apt-get install --force-yes -y \
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
 zip\
 git


RUN git clone -b modified2013_03 --single-branch https://github.com/ezdac/rdkit.git
ENV RDBASE=/rdkit
WORKDIR $RDBASE

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
RUN cmake -DRDK_BUILD_INCHI_SUPPORT=ON -DRDK_BUILD_AVALON_SUPPORT=ON AVALONTOOLS_DIR=$AVALONTOOLS_DIR ..
RUN make
RUN make install
#

# install dependencies for the benchmarking scripts
RUN apt-get install -y --force-yes \
python-pip \
python-setuptools

RUN apt-get install -y --force-yes \
libatlas-base-dev \
gcc \
gfortran \
g++

WORKDIR /
RUN pip install scipy==0.12.0
RUN pip install scikit-learn==0.13
RUN pip install --upgrade setuptools

ADD . /benchmark
WORKDIR /benchmark
