FROM debian:jessie

# update distribution
RUN apt-get update -y
RUN apt-get upgrade -y

# install general tools
RUN apt-get install -y apt-utils gcc make automake build-essential git

# install python2 and python3 including the scientific packages (excl. plotting)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python cython 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 cython3 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python-six python3-six
RUN apt-get install -y python-setuptools python3-setuptools python-pip
RUN apt-get install -y python-numpy python-scipy 
RUN apt-get install -y python3-numpy python3-scipy

# install LaTeX stuff
RUN apt-get install -y texlive-science texlive-latex-extra

# install python tools for style checking
RUN pip install autopep8 
RUN pip install pycodestyle
