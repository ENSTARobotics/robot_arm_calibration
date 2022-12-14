#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>

#include <random>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_calibration_sampling_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation 
    Robot robot;

    //Robot visual tools initialisation
    RobotVisualTools visualTools;

    //Retrieve calibration configurations
    ros::NodeHandle n;
    std::string samplingMethod;
    int samplingValue;

    if(!n.getParam("samplingMethod",samplingMethod))
    {
        ROS_ERROR("Unable to retrieve calibration configurations sampling method !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("samplingValue",samplingValue))
    {
        ROS_ERROR("Unable to retrieve calibration configurations sampling value !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    //Open sampled configurations file
    std::string sampledConfigurationsFileName;
    if(!n.getParam("sampledConfigurationsFileName",sampledConfigurationsFileName))
    {
        ROS_ERROR("Unable to retrieve sampled configurations file name !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    YAML::Node config;
    try
    {
        config = YAML::LoadFile(sampledConfigurationsFileName);
    }
    catch(const std::exception& e)
    {
        boost::filesystem::path path(sampledConfigurationsFileName);
        boost::filesystem::create_directories(path.parent_path());
        boost::filesystem::ofstream {path};

        config = YAML::LoadFile(sampledConfigurationsFileName);
    }

    if (config["sampledConfigurations"]) 
    {
        config.remove("sampledConfigurations");
    }

    if(samplingMethod == "naive")
    {
        ROS_INFO("Computing naively sampled calibration configurations...");
        std::vector<double> Q;
        std::vector<std::vector<double>> jointsLimits = robot.getJointsLimits();
        for(int i = 0; i < samplingValue; i++)
        {
            for(int j = 0; j < robot.getJointsNumber(); j++)
            {
                if(jointsLimits[j][0] - 2*M_PI*floor(jointsLimits[j][0]/(2*M_PI)) != jointsLimits[j][1] - 2*M_PI*floor(jointsLimits[j][1]/(2*M_PI)))
                {
                    Q.push_back(jointsLimits[j][0] + (jointsLimits[j][1] - jointsLimits[j][0])*i/(samplingValue - 1));
                }
                else
                {
                    Q.push_back(jointsLimits[j][0] + (jointsLimits[j][1] - jointsLimits[j][0])*i/(samplingValue));
                }
            }
        }

        for(int k = 0; k < pow(samplingValue,robot.getJointsNumber()); k++)
        {
            std::vector<double> currentConfiguration {Q.begin(), Q.begin() + robot.getJointsNumber()};

            ROS_INFO("Testing configuration %d/%d: ", k+1, (int)pow(samplingValue,robot.getJointsNumber()));
            for(int i = 0; i < currentConfiguration.size(); i++)
            {
                ROS_INFO("J%d : %f",i+1,currentConfiguration[i]);
            }

            if(robot.isReachable(currentConfiguration,true))
            {
                ROS_INFO("Reachable !");
                config["sampledConfigurations"].push_back(currentConfiguration);
            }
            else
            {
                ROS_INFO("Unreachable...");
            }
            
            for(int l = 0; l < robot.getJointsNumber(); l++)
            {
                if((k+1)%(int)pow(samplingValue,l) == 0)
                {
                    double tempValue = Q[l];
                    for(int m = 0; m < samplingValue - 1; m++)
                    {
                        Q[m*robot.getJointsNumber()+l] = Q[(m+1)*robot.getJointsNumber()+l];
                    }
                    Q[(samplingValue-1)*robot.getJointsNumber()+l] = tempValue;
                }
            }
        }
    }

    else if(samplingMethod == "scan")
    {
        double radiusMin = 0.2, radiusMax = 0.5, currentRadius;
        double inclinationMin = M_PI/6, inclinationMax = M_PI/3, currentInclination;
        double currentAzimuth;

        geometry_msgs::Pose currentPose, spherePose;
        spherePose.position.x = 0.3;
        spherePose.position.z = 0.1;
        visualTools.addSphere("collisionSphere",spherePose,0.1,false);

        std::vector<double> currentConfiguration;
        std::vector<geometry_msgs::Pose> sampledPoses;

        bool reachable = false;

        for(int i = 0; i < samplingValue; i++)
        {
            for(int j = 0; j < samplingValue; j++)
            {
                for(int k = 0; k < samplingValue; k++)
                {
                    currentRadius = radiusMin + (radiusMax - radiusMin)*i/(samplingValue - 1);
                    currentInclination = inclinationMin + (inclinationMax - inclinationMin)*j/(samplingValue - 1);
                    currentAzimuth = 0.0 + (2*M_PI)*k/samplingValue;

                    reachable = false;
                    for(int l = 0; l < 4; l++)
                    {   
                        currentPose = sphericPose(spherePose,currentRadius,currentInclination,currentAzimuth,true,l*M_PI/2);
                        if(l >= 1)
                        {
                            ROS_INFO("Trying with yaw offset !");
                        }
                        ROS_INFO("Testing pose %d/%d: ", i*samplingValue*samplingValue + j*samplingValue + k + 1, samplingValue*samplingValue*samplingValue);
                        ROS_INFO("X : %f, Y : %f, Z : %f", currentPose.position.x, currentPose.position.y, currentPose.position.z);

                        reachable = robot.isReachable(currentPose,currentConfiguration,true);
                        if(reachable)
                        {
                            break;
                        }
                    }

                    if(reachable)
                    {
                        ROS_INFO("Reachable !");
                        ROS_INFO("Corresponding configuration : ");
                        for(int u = 0; u < currentConfiguration.size(); u++)
                        {
                            ROS_INFO("J%d : %f",u+1,currentConfiguration[u]);
                        }
                        config["sampledConfigurations"].push_back(currentConfiguration);
                    }   
                    else
                    {
                        ROS_INFO("Unreachable...");
                    }
                }
            }
        }
    }

    else if(samplingMethod == "testing_naive")
    {
        ROS_INFO("Computing naive testing calibration configurations...");
        std::vector<std::vector<double>> jointsLimits = robot.getJointsLimits();
        std::vector<double> currentConfiguration;

        //Random configuration generation
        std::default_random_engine re;
        std::vector<std::uniform_real_distribution<double>> unifVector;
        std::uniform_real_distribution<double> unif;

        for(int j = 0; j < robot.getJointsNumber(); j++)
        {
            unif = std::uniform_real_distribution<double>(jointsLimits[j][0],jointsLimits[j][1]);
            unifVector.push_back(unif);
        }

        for(int i = 0; i < samplingValue; i++)
        {
            ROS_INFO("Trying to find testing configuration %d",i+1);
            for(int j = 0; j < robot.getJointsNumber(); j++)
            {
                ROS_INFO("%f,%f",jointsLimits[j][0],jointsLimits[j][1]);
                currentConfiguration.push_back(unifVector[j](re));
            }

            ROS_INFO("Testing configuration :");
            for(int k = 0; k < currentConfiguration.size(); k++)
            {
                ROS_INFO("J%d : %f",k+1,currentConfiguration[k]);
            }

            if(robot.isReachable(currentConfiguration,true))
            {
                ROS_INFO("Reachable !");
                config["targetConfigurations"].push_back(currentConfiguration);
            }
            else
            {
                ROS_INFO("Unreachable...");
                i--;
            }
            currentConfiguration.clear();
        }
    }

    else if(samplingMethod == "testing_scan")
    {
        ROS_INFO("Computing scan testing calibration configurations...");

        double radiusMin = 0.2, radiusMax = 0.5, currentRadius;
        double inclinationMin = M_PI/6, inclinationMax = M_PI/3, currentInclination;
        double currentAzimuth;

        //Random configuration generation
        std::default_random_engine re;
        std::uniform_real_distribution<double> unifRadius, unifInclination;

        unifRadius = std::uniform_real_distribution<double>(radiusMin,radiusMax);
        unifInclination = std::uniform_real_distribution<double>(inclinationMin,inclinationMax);
        currentRadius = unifRadius(re);
        currentInclination = unifInclination(re);

        geometry_msgs::Pose currentPose, spherePose;
        spherePose.position.x = 0.5;
        spherePose.position.z = 0.2;
        visualTools.addSphere("collisionSphere",spherePose,0.1,false);

        bool reachable;
        std::vector<double> currentConfiguration;

        for(int k = 0; k < samplingValue; k++)
        {
            currentAzimuth = 0.0 + (2*M_PI)*k/samplingValue;

            reachable = false;
            for(int l = 0; l < 4; l++)
            {   
                currentPose = sphericPose(spherePose,currentRadius,currentInclination,currentAzimuth,true,l*M_PI/2);
                if(l >= 1)
                {
                    ROS_INFO("Trying with yaw offset !");
                }
                ROS_INFO("Testing pose %d/%d: ", k + 1, samplingValue);
                ROS_INFO("X : %f, Y : %f, Z : %f", currentPose.position.x, currentPose.position.y, currentPose.position.z);

                reachable = robot.isReachable(currentPose,currentConfiguration,true);
                if(reachable)
                {
                    break;
                }
            }

            if(reachable)
            {
                ROS_INFO("Reachable !");
                ROS_INFO("Corresponding configuration : ");
                for(int u = 0; u < currentConfiguration.size(); u++)
                {
                    ROS_INFO("J%d : %f",u+1,currentConfiguration[u]);
                }
                config["sampledConfigurations"].push_back(currentConfiguration);
            }   
            else
            {
                ROS_INFO("Unreachable...");
            }
        }
    }

    //Close file
    std::ofstream fout(sampledConfigurationsFileName);   
    fout << config;

    ros::shutdown();
    return 0;
}

