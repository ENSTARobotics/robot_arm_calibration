#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>
#include <robot_arm_tools/RobotVisualTools.h>

#include <std_srvs/Empty.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_calibration_measurements_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    ros::NodeHandle n;

    //Robot initialisation 
    Robot robot;

    //Robot visual tools initialisation
    RobotVisualTools visualTools;

    //Move the robot to its initial configuration
    robot.init(false,true);

    //Retrieve calibration targets
    std::vector<std::vector<double>> waypoints;
    std::vector<double> currentConfiguration;
    XmlRpc::XmlRpcValue targetConfigurations;
    
    if(!n.getParam("targetConfigurations",targetConfigurations))
    {
        ROS_ERROR("Unable to retrieve robot calibration configurations !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    
    for (int i = 0; i < targetConfigurations.size(); i++)
    {
        for(int j = 0; j < targetConfigurations[i].size(); j++)
        {
            currentConfiguration.push_back(targetConfigurations[i][j]);
        }
        waypoints.push_back(currentConfiguration);
        currentConfiguration.clear();
    }

    //Main loop with initial measurement
    robot.runMeasurementRoutine(waypoints, true);

    ros::shutdown();
    return 0;
}

