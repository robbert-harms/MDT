FROM ubuntu:16.04

RUN mkdir -p /src
COPY containers/silent.cfg /src

# install dependencies 
RUN apt-get update && apt-get install -y lsb-core wget

# install Intel OpenCL runtime
RUN cd /src && \ 
    wget http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz && \
	tar -xvzf opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz && \
	mv silent.cfg opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25 && \
	cd opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25 && \
	./install.sh --silent silent.cfg --cli-mode

# install mdt
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:robbert-harms/cbclab
RUN apt-get update && apt-get install -y python3-mdt python3-pip
RUN pip3 install tatsu
