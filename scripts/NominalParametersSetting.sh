#!/bin/bash

Help()
{
    #Display help
    echo "Script setting up the nominal model parameters for a full serial robot geometric calibration"
    echo "parameters :"

    echo "\$1 = Robot name"
    echo "\$2 = Path to the robot URDF (.urdf) or DH parameters file (.py - see template in config/nominalModels/diverse/template.py)"
    echo "\$3 = Path to the robot base parameters file (.yaml) containing the robot base offset transformation regarding to the measuring device reference frame (no robot base measurements), or the measured robot base frame measurement"

    echo "\$4 = Marker name"
    echo "\$5 = Path to the markers parameters file (.py - see template in config/nominalModels/marker/template.py)"

    echo "\$6 = (optional) Robot flange link name"
}

#Help
while getopts ":h" option; do
   case $option in
      h) 
         Help
         exit;;
   esac
done

#Getting robot_arm_calibration folder
package_folder=$(rospack find robot_arm_calibration)

### ROBOT NOMINAL PARAMETERS

skip_init=0

#Create robot nominal parameters folder
if ! [ -d "$package_folder/config/nominalModels/$1" ]; then
    mkdir -p "$package_folder/config/nominalModels/$1"
fi

#Create robot nominal parameters importation python script (will be called via "import" later)
if ! [ -f "$package_folder/config/nominalModels/$1/$1_parameters.py" ]; then
    touch "$package_folder/config/nominalModels/$1/$1_parameters.py"
else
    #If file exists, overwrite...
    read -e -p "File $package_folder/config/nominalModels/$1/$1_parameters.py already exists, overwrite ? (y/n)" answer
    if [[ $answer =~ ^[Yy]$ ]]; then
        rm "$package_folder/config/nominalModels/$1/*.py"
        rm "$package_folder/config/nominalModels/$1/*.pickle"
        rm "$package_folder/config/nominalModels/$1/$1_parameters.py"
        touch "$package_folder/config/nominalModels/$1/$1_parameters.py"
    #...Or skip initialisation
    else
        skip_init=1
    fi
fi

#Get robot name and nominal parameters setting method
filename=$(basename -- "$2")
extension="${filename##*.}"

if [ $skip_init = 0 ]; then
    #Case where DH parameters are used
    #TODO DH from parameters file rather than from python script
    if [ "$extension" = "py" ]; then
        cat $2 >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
        echo -e "\n" >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
        import_routine="### Import routine\nfrom robot_arm_calibration.DHtoModel import DHtoModel\nJOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = DHtoModel(DHParameters)\n"
        echo -e $import_routine >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
    #Case where URDF/XACRO file is used
    elif [ "$extension" = "xacro" ] || [ "$extension" = "urdf" ]; then
        echo -e "urdfFilePath = '$2'\n" >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
        if [ -z "$6" ]; then
            $6=""
        fi
        echo -e "flangeLinkName = '$6'\n" >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
        import_routine="### Import routine\nfrom robot_arm_calibration.URDFtoModel import URDFtoModel\nJOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = URDFtoModel(urdfFilePath,flangeLinkName)\n"
        echo -e $import_routine >> "$package_folder/config/nominalModels/$1/$1_parameters.py"
    else
        echo "Invalid input format ! "
        exit 1
    fi
    #Create a local copy of the robot base parameters
    cp $3 "$package_folder/config/nominalModels/$1/$1_base_parameters.yaml"
fi

### MARKER NOMINAL PARAMETERS

#Create marker nominal parameters importation python script (will be called via "import" later)
if ! [ -f "$package_folder/config/nominalModels/markers/$4_parameters.py" ]; then
    #touch "$package_folder/config/nominalModels/markers/$4_parameters.py"
    cp $5 "$package_folder/config/nominalModels/markers/$4_parameters.py"
else
    #If file exists, overwrite or not
    read -e -p "File $package_folder/config/nominalModels/markers/$4_parameters.py already exists, overwrite ? (y/n) " answer
    if [[ $answer =~ ^[Yy]$ ]]; then
        #rm "$package_folder/config/nominalModels/markers/$4_parameters.py"
        #touch "$package_folder/config/nominalModels/markers/$4_parameters.py"
        cp $5 "$package_folder/config/nominalModels/markers/$4_parameters.py"
    fi
fi