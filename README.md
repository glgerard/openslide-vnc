# openslide-vnc
A docker containing a VNC desktop to run openslide apps and the ASAP application

It installs an Anaconda 3 environment with OpenCV 3.1 on Ubuntu 16.04. It also installs
all the dependencies for openslide and ASAP to work properly.

For optimal display performance users connect with a VNC client to port 5901 of their
localhost (or a port of their choice) and execute ASAP or their openslide based
applications on the desktop environment of the docker container.

## How to connect

The default VNC password is dockervnc

The recommended way to start the docker is with

```bash
  docker run -d --rm -v $PWD:/tmp/working -v <host-data-path>:/data -p 5901:5901 <image-name>
```

## Windows specific

If you are building the image on Windows before cloning the repo run the following

```bash
  git config core.eol lf
```

This will prevent the configuration files to be terminated with CRLF which would cause an issue once copied
in the container filesystem.
