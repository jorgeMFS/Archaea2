FROM ubuntu:20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update && apt install -y python3-pip wget

RUN apt-get update -y && apt-get install -y bc

RUN apt-get install -y bc

RUN apt-get install -y unzip

RUN apt-get install -y jq

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ADD . /Archaea

WORKDIR /Archaea

RUN bash run.sh

CMD tail -f >> /dev/null
