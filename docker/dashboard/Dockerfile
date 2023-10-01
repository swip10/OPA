FROM opa:latest

WORKDIR /code

COPY ./config/config_sql.ini /code/config/config_sql.ini
COPY ./start_scripts.sh /code/src/start_scripts.sh

RUN chmod +x /code/src/start_scripts.sh
CMD ["/code/src/start_scripts.sh"]