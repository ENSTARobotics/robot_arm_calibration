for f in $(pwd)/O1_greedy_50/TargetConfigurations.yaml
do
	roslaunch robot_arm_calibration Measurements.launch robot_name:=ur10e tool_name:=marker3D target_configurations_file:=$f measuring_device:=optitrack rosbag_name:=$(dirname $f)
done

#roslaunch robot_arm_calibration Measurements.launch robot_name:=ur10e tool_name:=marker3D target_configurations_file:="/home/demo1/Desktop/robot_arm_calibration/config/calibratedModels/ur10e_marker3D/random_20/SampledConfigurations.yaml" measuring_device:=optitrack rosbag_name:="/home/demo1/Desktop/robot_arm_calibration/config/calibratedModels/ur10e_marker3D/random_20/random_20"
