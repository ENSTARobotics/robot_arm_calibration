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

    echo "\$6 = Name of the measuring device as specified in the corresponding launch file"

    echo "\$7 = (optional) Path to the robot configurations sampling method description file for measurements - default : naive_sampling.yaml"
    echo "\$8 = (optional) Path to the robot configurations sampling method description file for verification - default : random_sampling.yaml"

    echo "\$9 = (optional) Robot configurations selection algorithm - default : random"

    echo "\$10 = (optional) Parameters identification algorithm - default : GN"

    echo "\$11 = (optional) Additionnal output - default : urdf"
}

#Help
while getopts ":h" option; do
   case $option in
      h)
         Help
         exit;;
   esac
done

#YAML parser
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|,$s\]$s\$|]|" \
        -e ":1;s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s,$s\(.*\)$s\]|\1\2: [\3]\n\1  - \4|;t1" \
        -e "s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s\]|\1\2:\n\1  - \3|;p" $1 | \
   sed -ne "s|,$s}$s\$|}|" \
        -e ":1;s|^\($s\)-$s{$s\(.*\)$s,$s\($w\)$s:$s\(.*\)$s}|\1- {\2}\n\1  \3: \4|;t1" \
        -e    "s|^\($s\)-$s{$s\(.*\)$s}|\1-\n\1  \2|;p" | \
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)-$s[\"']\(.*\)[\"']$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)-$s\(.*\)$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" | \
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]; idx[i]=0}}
      if(length($2)== 0){  vname[indent]= ++idx[indent] };
      if (length($3) > 0) {
         split($3,tab,"#");
         gsub(/ /,"",tab[1])
         vn=""; for (i=0; i<indent; i++) { vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, vname[indent], tab[1]);
      }
   }'
}

### ROBOT/MARKER NOMINAL PARAMETERS

#Retrieve robot flange link name for URDF handling
roslaunch robot_arm_tools Setup_$1.launch --dump-params | sed -e '/"</,/n"/d' > tmp.yaml
cut -c 2- tmp.yaml > robot_default_params.yaml
rm tmp.yaml
eval $(parse_yaml robot_default_params.yaml)
rm robot_default_params.yaml

#Setup robot/marker nominal parameters files
rosrun robot_arm_calibration NominalParametersSetting.sh $1 $2 $3 $4 $5 ${flangeLinkName}|| exit 1
#Also works : bash NominalParametersSetting.sh $1 $2 $3 $4 $5 ${flangeLinkName}|| exit 1

### MODELING

python3 -m robot_arm_calibration.AnalyticGeometricModel $1 $4
cd "$package_folder/config/calibratedModels/$1_$4"

### SAMPLING

#For identification

#Check for sampling method file
if [ -z "$7" ]; then
    $7="$package_folder/config/sampling/naive_sampling.yaml"
fi

roslaunch robot_arm_calibration Sampling.launch robot_name:=$1 tool_name:=$4 sampling_parameters_file:=$7    

#Create storage folder depending on sampling parameters
eval $(parse_yaml $7)

sampling_folder_measurements= "${samplingMethod}_${samplingValue}"
if [ -z ${cartesianLimits+x} ]; then
    sampling_folder_measurements="${sampling_folder_measurements}_box"
elif [ -z ${sphericLimits+x} ]; then
    sampling_folder_measurements="${sampling_folder_measurements}_scan"
fi

#For verification

#Check for sampling method file
if [ -z "$8" ]; then
    $8="$package_folder/config/sampling/random_sampling.yaml"
fi

roslaunch robot_arm_calibration Sampling.launch robot_name:=$1 tool_name:=$4 sampling_parameters_file:=$8    

#Create storage folder depending on sampling parameters
eval $(parse_yaml $8)

sampling_folder_verification= "${samplingMethod}_${samplingValue}"
if [ -z ${cartesianLimits+x} ]; then
    sampling_folder_verification="${sampling_folder_verification}_box"
elif [ -z ${sphericLimits+x} ]; then
    sampling_folder_verification="${sampling_folder_verification}_scan"
fi

### MEASUREMENTS CONFIGURATIONS CHOICE

cd sampling_folder_measurements

#Check for choice method
if [ -z "$9" ]; then
    $9="random"
fi

python3 -m robot_arm_calibration.TargetConfigurationsGeneration $9
cd "$9"

### MEASUREMENTS

#For identification
roslaunch robot_arm_calibration Measurements.launch robot_name:=$1 tool_name:=$4 target_configurations_file:="$package_folder/config/calibratedModels/$1_$4/$sampling_folder_measurements/$9/TargetConfigurations.yaml" measuring_device:=$6

#For verification
roslaunch robot_arm_calibration Measurements.launch robot_name:=$1 tool_name:=$4 target_configurations_file:="$package_folder/config/calibratedModels/$1_$4/$sampling_folder_verification/SampledConfigurations.yaml" measuring_device:=$6

### IDENTIFICATION

cd "measurements"

#Check for identification method
if [ -z "$10" ]; then
    $10="GN"
fi

python3 -m robot_arm_calibration.ParametersIdentification $10

### VERIFICATION

cd "$package_folder/config/calibratedModels/$1_$4"
python3 -m robot_arm_calibration.ResultsDisplay

### OUTPUTS GENERATION

if [ -z "$11" ]; then
    if [ "$extension" = "py" ]; then
        read -e -p "URDF description file path ? " URDF_path
    else
        URDF_path=$2

    python3 -m robot_arm_calibration.ModeltoURDF "$package_folder/config/calibratedModels/$1_$4/$(basename $7)/$8/measurements/$9/CalibratedModel.yaml" $URDF_path ${flangeLinkName}
fi