FROM python:3.8

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./src /code/src
COPY ./setup.py /code/setup.py
COPY ./README.md /code/README.md
RUN pip install -e /code/.
# RUN python setup.py install

COPY ./models /code/models
COPY ./config /code/config
COPY ./tests /code/tests
