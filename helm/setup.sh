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

################################################################################
# Run                                                                         #
################################################################################
Run()
{
    helm install postgres postgresql-chart/ -f postgresql-chart/values.yaml -f postgresql-chart/values_ber.yaml
    helm install mongodb mongodb-chart/ -f mongodb-chart/values.yaml -f postgresql-chart/values_ber.yaml
    sleep 60
    helm install dashboard dashboard-chart/ -f dashboard-chart/values.yaml
    helm install fastapi fastapi-chart/ -f fastapi-chart/values.yaml
    sleep 10
}



################################################################################
# Stop                                                                         #
################################################################################
Stop()
{ 
    helm uninstall dashboard
    sleep 30
    helm uninstall fastapi
    sleep 30
    helm uninstall mongodb
    sleep 30
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
