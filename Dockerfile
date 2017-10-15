FROM ubuntu:16.04

WORKDIR /tmp/working

# Set proxy server, replace host:/port with values for your servers
#ENV http_proxy 'http://<your-proxy-ip>:<port>'
#ENV https_proxy 'https://<your-proxy-ip>:<port>'

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libgles2-mesa
RUN apt-get install -y wget bzip2 gcc libtasn1-3-bin xfce4 xfce4-goodies

# Install TigerVNC

RUN wget --quiet https://bintray.com/artifact/download/tigervnc/stable/ubuntu-16.04LTS/amd64/tigervncserver_1.8.0-1ubuntu1_amd64.deb && \
    dpkg -i tigervncserver_1.8.0-1ubuntu1_amd64.deb && \
    apt-get -f install && \
    rm tigervncserver_1.8.0-1ubuntu1_amd64.deb

EXPOSE 5901

# Configure the VNC server

ADD vnc.tar.bz2 /root

RUN chmod og+x /root/.vnc/xstartup
RUN chmod 0600 /root/.vnc/passwd
RUN chmod -x /root/.vnc/config

# Install Openslide

#RUN apt-get install -y openslide-tools

# Use a custom build as the official package
# has a bug on Ubuntu 16.04 such that the lower
# sampled dimensions can not be reported correctly

RUN wget --quiet https://github.com/glgerard/openslide-vnc/releases/download/v0.1/openslide_3.4-1_amd64.deb && \
    dpkg -i openslide_3.4-1_amd64.deb && \
    apt-get -y install libopenjpeg5 && \
    rm openslide_3.4-1_amd64.deb

# Install ASAP

# Install ASAP pre-requisites

RUN apt-get install -y libboost-dev libboost-program-options1.58.0 libboost-regex1.58.0 libboost-thread1.58.0 \
                       libdcmtk5 libpugixml1v5 libpython3.5 libqt5core5a libqt5gui5-gles libqt5widgets5 libjasper1

RUN wget --quiet https://github.com/GeertLitjens/ASAP/releases/download/1.7.3/ASAP-1.7-Linux-python35.deb && \
    dpkg -i ASAP-1.7-Linux-python35.deb && \
    rm ASAP-1.7-Linux-python35.deb

ENV PATH /opt/ASAP/bin:/opt/openslide/bin:$PATH
   
ENV LD_LIBRARY_PATH /opt/ASAP/bin:/opt/openslide/lib:$LD_LIBRARY_PATH
 
# Install Anaconda3

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh -O anaconda3.sh && \
    /bin/bash anaconda3.sh -b -p /opt/conda && \
    rm anaconda3.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda update --all && pip install openslide-python && conda install -y -c menpo opencv3

CMD [ "vncserver", "-fg" ]