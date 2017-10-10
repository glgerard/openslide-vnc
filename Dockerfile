FROM ubuntu:16.04

WORKDIR /tmp/working

# Set proxy server, replace host:/port with values for your servers
#ENV http_proxy 'http://<your-proxy-ip>:<port>'
#ENV https_proxy 'https://<your-proxy-ip>:<port>'

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y wget bzip2 gcc libtasn1-3-bin xfce4 xfce4-goodies

# Install Anaconda3

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh -O anaconda3.sh && \
    /bin/bash anaconda3.sh -b -p /opt/conda && \
    rm anaconda3.sh

# Install TigerVNC

RUN wget --quiet https://bintray.com/artifact/download/tigervnc/stable/ubuntu-16.04LTS/amd64/tigervncserver_1.8.0-1ubuntu1_amd64.deb && \
    dpkg -i tigervncserver_1.8.0-1ubuntu1_amd64.deb && \
    apt-get -f install && \
    rm tigervncserver_1.8.0-1ubuntu1_amd64.deb

ENV PATH /opt/conda/bin:$PATH

RUN apt-get install -y openslide-tools

RUN conda update --all && pip install openslide-python && conda install -y -c menpo opencv3

RUN apt-get install -y libboost-dev libboost-program-options1.58.0 libboost-regex1.58.0 libboost-thread1.58.0 \
                       libdcmtk5 libpugixml1v5 libpython3.5 libqt5core5a libqt5gui5-gles libqt5widgets5 libjasper1

RUN wget --quiet https://github.com/GeertLitjens/ASAP/releases/download/1.7.3/ASAP-1.7-Linux-python35.deb && \
    dpkg -i ASAP-1.7-Linux-python35.deb && \
    rm ASAP-1.7-Linux-python35.deb

ENV PATH /opt/ASAP/bin:$PATH
    
EXPOSE 5901

RUN mkdir /root/.vnc
ADD xstartup /root/.vnc
ADD config /root/.vnc
ADD passwd /root/.vnc

# Necessary if the container is built on Windows
# hosts
RUN chmod og+x /root/.vnc/xstartup
RUN chmod 0600 /root/.vnc/passwd
RUN chmod -x /root/.vnc/config

#CMD [ "/bin/bash" ]
CMD [ "vncserver", "-fg" ]
