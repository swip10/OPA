# OPA

Welcome to the OPA projet ! 
The goal is to build a pipeline to automatically process cryptomoney stock exchange and train a model to put and call orders. 

## installation 

### configuration API keys 

At the root of the projet at folder name "config" and create a file config.ini inside.  
The config/config.ini file should contain the following info:

```
[API]
BINANCE_API_KEY = 'put your api key here'
BINANCE_API_SECRET = 'put the api scret here'
```

### python packages 

Install the site-packages using pip with the following command 

```
$ cd src
$ pip install requirements.txt
```

