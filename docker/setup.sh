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
    cd ./docker
}


################################################################################
# Run                                                                         #
################################################################################
Run()
{
    cd ./mongodb
    docker-compose up -d 
    cd ./../postgresql
    docker-compose up -d 
    cd ..
}


################################################################################
# Stop                                                                         #
################################################################################
Stop()
{
    cd ./mongodb
    docker-compose down
    cd ./../postgresql
    docker-compose down 
    cd ..
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
while getopts ":h" option; do
   case $option in
    case $option in
      a) # display Help
         Build
         Run
         exit;;
    case $option in
      b) # display Help
         Build
         exit;;
      h) # display Help
         Help
         exit;;
    case $option in
      s) # display Help
         Stop
         exit;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done

echo "Hello world!"
