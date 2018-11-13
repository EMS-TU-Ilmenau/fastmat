FROM debian:jessie

# update distribution
RUN \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-utils gcc make automake build-essential git && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python cython python-six \
        python3 cython3 python3-six \
        python-setuptools python3-setuptools python-pip \
        python-numpy python-scipy \
        python3-numpy python3-scipy && \
    apt-get install -y --no-install-recommends \
        texlive-science texlive-latex-extra && \
    apt-get -y autoremove && \
    apt-get -y autoclean && \
    apt-get -y clean all && \
    pip install autopep8==1.3.3 pycodestyle==2.3.1 && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt && \
    rm -rf /root/.cache/pip && \
    rm -rf /root/.pip/cache && \
    rm -rf /tmp/*
