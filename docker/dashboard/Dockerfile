FROM opa:latest

WORKDIR /code

COPY ./config/config_sql.ini /code/config/config_sql.ini

# Définir le fuseau horaire
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN chmod +x /code/src/start_scripts.sh
CMD ["/code/src/start_scripts.sh"]