#!/bin/bash

robot_config_path="$../config/calibratedModels/$1_$2/*/"

for d in $robot_config_path ; do
    if [ -d "$d" ] ; then

        echo "Folder : $(basename $d)"
    
        # Testing folder
        if [[ "$(basename $d)" == *"testing"* ]]; then
            configurations_file=$(ls $d*.yaml)
            echo "Running measurements with $(basename $configurations_file) !"
            roslaunch robot_arm_tools CalibrationMeasurements.launch robot_name:=$1 tool_name:=$2 safety_distance:=$3 target_configurations_file:=$configurations_file simulation:=false || { echo "Execution failed !" ; exit 1; }
        
        # Identification folder
        else
            folder_path="$d*/"
            for sd in $folder_path ; do
                if [ -d "$sd" ] && [ $(basename $sd) != "measurements" ]; then
                    echo "Sub-folder : $(basename $sd)"
                    configurations_file=$(ls $sd*.yaml)
                    echo "Running measurements with $(basename $configurations_file) !"
                    roslaunch robot_arm_tools CalibrationMeasurements.launch robot_name:=$1 tool_name:=$2 safety_distance:=$3 target_configurations_file:=$configurations_file simulation:=false || { echo "Execution failed !" ; exit 1; }
                fi
            done
        fi
    fi
done
