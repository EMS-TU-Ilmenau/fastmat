FROM debian:jessie
LABEL maintainer="christoph.wagner@tu-ilmenau.de"

# update distribution
RUN apt-get update -y

# install general tools
RUN apt-get install -y --no-install-recommends \
    apt-utils=1.4.8 gcc=6.3.0-4 make=4.1-9.1 automake=1.15-6 \
    build-essential=12.3 git=2.11.0-3+deb9u2

# install python2 and python3 including the scientific packages (excl. plotting)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no.install-recommends \
    python=2.7.13-2 cython=0.25.2-1 python-six=1.10.0-3 \
    python3=3.5.3-1 cython3=0.25.2-1 python3-six=1.10.0-3 \
    python-setuptools=33.1.1-1 python3-setuptools=33.1.1-1 python-pip=9.0.1-2 \
    python-numpy=1.12.1-3 python-scipy=0.18.1-2 \
    python3-numpy=1.12.1-3 python3-scipy=0.18.1-2

# install LaTeX stuff
RUN apt-get install -y --no-install-recommends \
    texlive-science=2016.20170123-5
    texlive-latex-extra=2016.20170123-5

# install python tools for style checking
RUN pip install autopep8=1.3.3 pycodestyle=2.3.1

# clean up
RUN apt-get clean
