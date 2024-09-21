FROM python:3.11

RUN apt update -y

WORKDIR /build-scratch
RUN apt install -y r-base libudunits2-dev gdal-bin libgdal-dev
RUN echo "install.packages('INLA',repos=c(getOption('repos'),INLA='https://inla.r-inla-download.org/R/stable'), dep=TRUE)" > install.R
RUN R --no-save < install.R

RUN pip install poetry

WORKDIR /code
COPY poetry.lock pyproject.toml /code/
RUN poetry install --with plotting --no-root
RUN mkdir plots

COPY . /code
RUN poetry install --with plotting

ENV SHELL=/bin/bash
CMD ["poetry", "shell"]
