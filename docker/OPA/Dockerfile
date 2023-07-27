FROM python:3.8

WORKDIR /code

COPY ./src /code/src
COPY ./requirements.txt /code/requirements.txt
COPY ./setup.py /code/setup.py 
COPY ./README.md /code/README.md
COPY ./models /code/models
COPY ./tests /code/tests

RUN pip install -r /code/requirements.txt
RUN pip install -e /code/.

