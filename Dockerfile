FROM yeop2/dalle-server:1

RUN \
    apt-get update && \
    apt-get install -y gcc &&\
    apt-get install -y g++
RUN apt-get install -y git

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD python app.py
