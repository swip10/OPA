#!/bin/bash

################################################################################
# Help                                                                         #
################################################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: scriptTemplate [-g|h|t|v|V]"
   echo "options:"
   echo "a     All - build and run docker images"
   echo "c     Stop containers and clean docker images"
   echo "b     Build OPA docker image."
   echo "h     Print this Help."
   echo "n     Create network."
   echo "r     Run docker images"
   echo "s     Stop docker images"
   echo "v     Verbose mode."
   echo "V     Print software version and exit."
   echo
}


################################################################################
# Build                                                                         #
################################################################################
Build()
{
    cd ..
    docker image build -t opa:latest -f docker/OPA/Dockerfile .
    docker image build -t opa/dashboard:latest -f docker/dashboard/Dockerfile .
    docker image build -t opa/fastapi:latest -f docker/fastAPI/Dockerfile .
    cd ./docker
    echo "removing <none> images "
    docker rmi -f $(docker images -f "dangling=true" -q)
}


################################################################################
# Network                                                                         #
################################################################################
Network()
{
    docker network create --subnet 172.50.0.0/16 --gateway 172.50.0.1 my_network
}

RemoveNetwork()
{
    docker network rm my_network
}

################################################################################
# Run                                                                         #
################################################################################
Run()
{
    Network
    cd ./mongodb
    docker-compose up -d 
    cd ./../postgresql
    docker-compose up -d 
    cd ./../fastAPI
    docker-compose up -d 

    # Attendre que le conteneur "postgres" soit prêt
    echo "Waiting for postgres to be ready..."
    until docker exec postgres pg_isready -h localhost -p 5432 -q; do
        sleep 1
    done
    echo "Postgres is ready!"

    cd ./../dashboard
    docker-compose up -d 
    cd ..
}


################################################################################
# Stop                                                                         #
################################################################################
Stop()
{
    echo "Stop containers" 
    cd ./mongodb
    docker-compose down
    cd ./../postgresql
    docker-compose down 
    cd ./../dashboard
    docker-compose down 
    cd ./../fastAPI
    docker-compose down 
    cd ..
    RemoveNetwork
}

################################################################################
################################################################################
# Main program                                                                 #
################################################################################
################################################################################
################################################################################
# Process the input options. Add options as needed.                            #
################################################################################
# Get the options
while getopts ":abnrsh:" option; do
   case $option in
      a) # display Help
         Build
         Run
         exit;;
      b) # display Help
         Build
         exit;;
      n) # create network
         Network
         exit;;
      r) # display Help
         Run
         exit;;
      s) # display Help
         Stop
         exit;;
      h) # display Help
         Help
         exit;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done

echo "End launcher OPA !"
