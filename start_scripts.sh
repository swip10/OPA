#!/bin/sh

# Exécuter dashboard.py en arrière-plan
python ./src/plots/dashboard.py &

# Exécuter postgreSQL_stream_script.py en arrière-plan
python ./src/postgreSQL_stream_script.py &

# Attendre la fin de tous les processus d'arrière-plan
wait
