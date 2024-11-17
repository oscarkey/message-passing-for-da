FROM python:3.11

RUN apt update -y

WORKDIR /build-scratch
RUN apt install -y r-base=4.2.2.20221110-2 libudunits2-dev=2.2.28-5 gdal-bin=3.6.2+dfsg-1+b2 libgdal-dev=3.6.2+dfsg-1+b2
RUN echo "install.packages('INLA',verison='24.6.27',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=TRUE)" > install.R
RUN R --no-save < install.R

RUN pip install poetry==1.8.4

WORKDIR /code
COPY poetry.lock pyproject.toml /code/
RUN poetry install --with plotting --no-root
RUN mkdir plots

COPY . /code
RUN poetry install --with plotting

ENV SHELL=/bin/bash
CMD ["poetry", "shell"]
