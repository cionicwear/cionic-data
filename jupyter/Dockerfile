# syntax = docker/dockerfile:1.0-experimental

#
# Add ffmpeg for videos
#
FROM node:alpine3.17
RUN apk add --no-cache ffmpeg python3 python3-dev gcc gfortran musl-dev
FROM jupyter/tensorflow-notebook:lab-4.0.2

#
# Install any necessary dependencies
#
ADD requirements.txt .
RUN pip3 install -r requirements.txt

#
# Add config to the container and update the paths
#
ADD jupyter_notebook_config.py /opt/conda/etc/jupyter/.

