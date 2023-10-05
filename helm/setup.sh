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
   echo "r     Run helm installs"
   echo "s     Stop helm"
   echo "v     Verbose mode."
   echo "V     Print software version and exit."
   echo
}


sleepy()
{
    if [ ! `whoami` = 'loic' ]
    then
      sleep $1
    fi
}


initialize()
{
    if [ `whoami` = 'loic' ]
    then
        export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
    fi
}

################################################################################
# Run                                                                         #
################################################################################
Run()
{
    initialize
    helm install postgres postgresql-chart/ -f postgresql-chart/values.yaml -f postgresql-chart/values_`whoami`.yaml
    helm install mongodb mongodb-chart/ -f mongodb-chart/values.yaml -f mongodb-chart/values_`whoami`.yaml
    sleepy 60
    helm install dashboard dashboard-chart/ -f dashboard-chart/values.yaml
    helm install fastapi fastapi-chart/ -f fastapi-chart/values.yaml
    sleepy 10
}



################################################################################
# Stop                                                                         #
################################################################################
Stop()
{
    initialize
    helm uninstall dashboard
    sleepy 15
    helm uninstall fastapi
    sleepy 15
    helm uninstall mongodb
    sleepy 15
    helm uninstall postgres
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
while getopts ":rsh:" option; do
   case $option in
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
