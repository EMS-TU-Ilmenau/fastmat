FROM debian:jessie
LABEL maintainer="christoph.wagner@tu-ilmenau.de"

# update distribution
RUN apt-get update -y && apt-get upgrade -y

# install general tools
RUN apt-get install -y --no-install-recommends \
    apt-utils gcc make automake build-essential git

# install python2 and python3 including the scientific packages (excl. plotting)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no.install-recommends \
    python cython python-six \
    python3 cython3 python3-six \
    python-setuptools python3-setuptools python-pip \
    python-numpy python-scipy \
    python3-numpy python3-scipy

# install LaTeX stuff
RUN apt-get install -y --no-install-recommends \
    texlive-science texlive-latex-extra

# install python tools for style checking
RUN pip install autopep8=1.3.3 pycodestyle=2.3.1

# clean up
RUN apt-get clean
