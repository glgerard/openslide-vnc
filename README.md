# openslide-vnc
A docker containing a VNC desktop to run openslide apps and the ASAP application

It installs an Anaconda 3 environment with OpenCV 3.1 on Ubuntu 16.04. It also installs
all the dependencies for openslide and ASAP to work properly.

For optimal display performance users connect with a VNC client to port 5901 of their
localhost (or a port of their choice) and execute ASAP or their openslide based
applications on the desktop environment of the docker container.
