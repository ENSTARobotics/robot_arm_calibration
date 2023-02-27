#!/bin/bash

Help()
{
    #Display help
    echo "Script launching a full serial robot geometric calibration"
    echo "parameters :"

    echo "\$1 = Robot name"
    echo "\$2 = Path to the robot URDF (.urdf) or DH parameters file (.py - see template in config/nominalModels/diverse/template.py)"
    echo "\$3 = Path to the robot base parameters file (.yaml) containing the robot base offset transformation regarding to the measuring device reference frame (no robot base measurements), or the measured robot base frame measurement"

    echo "\$4 = Marker name"
    echo "\$5 = Path to the markers parameters file (.py - see template in config/nominalModels/marker/template.py)"
}

#Getting robot_arm_calibration folder
package_folder=$(rospack find robot_arm_calibration)

#TODO DH from parameters file !
### Inputs

#Create parameters folder

skip_init=0

if ! [ -d "$package_folder/config/nominalModels/$1" ]; then
    mkdir "$package_folder/config/nominalModels/$1"
fi

if ! [ -f "$package_folder/config/nominalModels/$1/$1_parameters.py" ]; then
    touch "$package_folder/config/nominalModels/$1/$1_parameters.py"
else
    read -p "File $package_folder/config/nominalModels/$1/$1_parameters.py already exists, overwrite ? y/n " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$package_folder/config/nominalModels/$1/*.py"
        rm "$package_folder/config/nominalModels/$1/*.pickle"
        touch "$package_folder/config/nominalModels/$1/$1_parameters.py"
    else
        skip_init=1
    fi
fi

filename=$(basename -- "$2")
extension="${filename##*.}"

if [ skip_init = 0 ]; then
    if [ "$extension" = "py" ]; then
        cat $2 >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
        echo -e "\n" >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
        import_routine="### Import routine\nfrom robot_arm_calibration.DHtoModel import DHtoModel\nJOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = DHtoModel(DHParameters)\n"
        echo -e $import_routine >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
    elif [ "$extension" = "xacro" ] || [ "$extension" = "urdf" ]; then
        echo -e "urdfFilePath = '$2'\n"
        import_routine="### Import routine\nfrom robot_arm_calibration.URDFtoModel import URDFtoModel\nJOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = URDFtoModel(urdfFilePath)\n"
        echo -e $import_routine >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
    else
        echo "Invalid input format ! "
        exit 1
    fi
    cp $3 "$package_folder/config/nominalModels/$1/$1_base_parameters.yaml"
fi

if ! [ -f "$package_folder/config/nominalModel/markers/$4_parameters.py" ]; then
    #touch "$package_folder/config/nominalModels/markers/$4_parameters.py"
    cp $5 "$package_folder/config/nominalModels/markers/$4_parameters.py"
else
    read -p "File $package_folder/config/nominalModels/markers/$4_parameters.py already exists, overwrite ? y/n " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        #rm "$package_folder/config/nominalModels/markers/$4_parameters.py"
        #touch "$package_folder/config/nominalModels/markers/$4_parameters.py"
        cp $5 "$package_folder/config/nominalModels/markers/$4_parameters.py"
    fi
fi