FROM continuumio/anaconda3


USER root
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    && apt-get -y install git \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && mkdir /root/codenation \
    && curl https://s3-us-west-1.amazonaws.com/codenation-cli/latest/codenation_linux.tar.gz | tar xvz \
    && mv codenation /usr/local/bin


EXPOSE 8888
