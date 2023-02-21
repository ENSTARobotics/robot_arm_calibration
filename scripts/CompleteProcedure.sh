#!/bin/bash

#$1 = Path to the robot URDF (.urdf) or DH parameters file (.py - see template in config/nominalModels/diverse_robots/template_parameters.py)
#$2 = Path to the markers parameters file (.py - see template in config/nominalModels/marker/template_parameters.py)
#$3 = (optional) Path to the configurations sampling parameters description file - default : naive_sampling.yaml
#$4 = (optional) Measurements configurations selection algorithm - default : random
#$5 = (optional) Parameters identification algorithm - default : GN
#$6 = (optional) Additionnal output - default : urdf

Help()
{
    #Display help
    echo "Script launching a full serial robot geometric calibration"
    echo "parameters :"
    echo "\$1 = Path to the robot URDF (.urdf) or DH parameters file (.py - see template in config/nominalModels/diverse/template.py)"
    echo "\$2 = Path to the markers parameters file (.py - see template in config/nominalModels/marker/template.py)"
    echo "\$3 = (optional) Path to the angular configurations sampling method description file - default : even_sampling.yaml"
    echo "\$4 = (optional) Measurements configurations selection algorithm - default : random"
    echo "\$5 = (optional) Parameters identification algorithm - default : GN"
    echo "\$6 = (optional) Additionnal output - default : urdf"
}

#Inputs


#Modeling


#Sampling


#Measurements design


#Measurements


#Identification


#Verification


#Outputs