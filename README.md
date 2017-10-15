# openslide-vnc
A docker containing a VNC desktop to run openslide apps and the ASAP application

It installs an Anaconda 3 environment with OpenCV 3.1 on Ubuntu 16.04. It also installs
all the dependencies for openslide and ASAP to work properly.

For optimal display performance users connect with a VNC client to port 5901 of their
localhost (or a port of their choice) and execute ASAP or their openslide based
applications on the desktop environment of the docker container.

## How to build

Clone the repository and from the directory containing the Dockerfile run

```bash
  docker build -t openslide-vnc .
```

## How to connect

The default VNC password is dockervnc

The recommended way to start the docker is with

```bash
  docker run -d --rm -v $PWD:/tmp/working -v <host-data-path>:/data -p 5901:5901 <image-name>
```

## Acknowledgements

OpenSlide is a C library for reading whole slide image files.
It has been developed at Carnegie Mellon University.
For further information please refer to http://openslide.org .

ASAP is an open source platform for visualizing, 
annotating and automatically analyzing whole-slide histopathology images.
For further information please refer to https://github.com/GeertLitjens/ASAP .