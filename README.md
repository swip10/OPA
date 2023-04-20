# OPA

Welcome to the OPA projet ! 
The goal is to build a pipeline to automatically process cryptomoney stock exchange and train a model to put and call orders. 

## installation 

### configuration API keys 

At the root of the projet at folder name "config" and create a file config.ini inside.  
The config/config.ini file should contain the following info:

```
[API]
BINANCE_API_KEY = put_your_api_key_here
BINANCE_API_SECRET = put_the_api_scret_here
```

### python packages 

Install the site-packages using pip with the following command 

```
$ cd src
$ pip install -r requirements.txt
```

### mongoDB

Install docker-compose and launch the service using the following script

```
$ cd docker/mongodb
$ mkdir db
$ docker-compose up -d 
```

### postgres 

Install docker-compose and launch the service using the following script

```
$ cd docker/postgresql
$ docker-compose up -d 
```