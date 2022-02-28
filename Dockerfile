FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN \
    apt-get update && \
    apt-get install -y gcc &&\
    apt-get install -y g++
RUN apt-get install -y git
RUN apt-get install -y nodejs

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD python app.py
