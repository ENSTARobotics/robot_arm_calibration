#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

#include <yaml-cpp/yaml.h>
#include <experimental/filesystem>

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

    std::string sampledConfigurationsStorageFolder;
    if(!n.getParam("sampledConfigurationsStorageFolder",sampledConfigurationsStorageFolder))
    {
        ROS_ERROR("Unable to retrieve sampled configurations folder name !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    sampledConfigurationsStorageFolder = sampledConfigurationsStorageFolder + samplingMethod + "_" + std::to_string(samplingValue);
    if(n.hasParam("cartesianLimits"))
    {
        sampledConfigurationsStorageFolder = sampledConfigurationsStorageFolder + "_box/";
        
    }
    else if(n.hasParam("sphericLimits"))
    {
        sampledConfigurationsStorageFolder = sampledConfigurationsStorageFolder + "_scan/";
    }
    else
    {
        sampledConfigurationsStorageFolder = sampledConfigurationsStorageFolder + "/";
    }

    //Handle sampled configurations storage folder - clean or create
    std::experimental::filesystem::path folderPath{sampledConfigurationsStorageFolder};
    if(std::experimental::filesystem::exists(folderPath))
    {
        ROS_WARN("%s already exists : its contents will be overwritten !", sampledConfigurationsStorageFolder.c_str());
    }
    else
    {
        ROS_INFO("Creating %s ...", sampledConfigurationsStorageFolder.c_str());
        std::experimental::filesystem::create_directories(folderPath.parent_path());
    }

    std::string sampledConfigurationsFileName = sampledConfigurationsStorageFolder + "SampledConfigurations.yaml";

    YAML::Node config;
    try
    {
        config = YAML::LoadFile(sampledConfigurationsFileName);
    }
    catch(const std::exception& e)
    {
        std::experimental::filesystem::path filePath(sampledConfigurationsFileName);
        std::ofstream {filePath};

        config = YAML::LoadFile(sampledConfigurationsFileName);
    }

    if (config["sampledConfigurations"]) 
    {
        config.remove("sampledConfigurations");
    }

    ROS_INFO("Computing sampled calibration configurations with %s method",samplingMethod.c_str());

    if(n.hasParam("cartesianLimits"))
    {
        //ROS_INFO("Adding cartesian constraints !");
        ROS_ERROR("Not yet implemented, soon !");
        throw std::logic_error("NOT IMPLEMENTED");
    }

    else if(n.hasParam("sphericLimits"))
    {
        ROS_INFO("Adding spheric constraints !");

        //Get sphere center
        geometry_msgs::Pose spherePose;
        std::vector<double> tmp;
        n.getParam("sphericLimits/center",tmp);
        spherePose.position.x = tmp[0];
        spherePose.position.y = tmp[1];
        spherePose.position.z = tmp[2];

        //Get radius, inclination and azimuth limits
        std::vector<double> radiusLimits;
        n.getParam("sphericLimits/radius",radiusLimits);
        std::vector<double> inclinationLimits;
        n.getParam("sphericLimits/inclination",inclinationLimits);
        std::vector<double> azimuthLimits;
        n.getParam("sphericLimits/azimuth",azimuthLimits);

        //Get orientation parameter
        bool inward;
        n.getParam("sphericLimits/inward",inward);

        double radiusMin = 0.2, radiusMax = 0.5, currentRadius;
        double inclinationMin = M_PI/6, inclinationMax = M_PI/3, currentInclination;
        double currentAzimuth;

        geometry_msgs::Pose currentPose;
        std::vector<double> currentConfiguration;
        std::vector<geometry_msgs::Pose> sampledPoses;

        bool reachable = false;

        if(samplingMethod == "naive")
        {
            for(int i = 0; i < samplingValue; i++)
            {
                if(radiusLimits[0] == radiusLimits[1]){i = samplingValue-1;}

                for(int j = 0; j < samplingValue; j++)
                {

                    if(inclinationLimits[0] == inclinationLimits[1]){j = samplingValue-1;}

                    for(int k = 0; k < samplingValue; k++)
                    {

                        if(azimuthLimits[0] == azimuthLimits[1]){k = samplingValue-1;}

                        currentRadius = radiusLimits[0] + (radiusLimits[1] - radiusLimits[0])*i/(samplingValue - 1);
                        currentInclination = inclinationLimits[0] + (inclinationLimits[1] - inclinationLimits[0])*j/(samplingValue - 1);
                        currentAzimuth = azimuthLimits[0] + (azimuthLimits[1] - azimuthLimits[0])*k/samplingValue;

                        reachable = false;
                        for(int l = 0; l < 4; l++)
                        {   
                            currentPose = sphericPose(spherePose,currentRadius,currentInclination,currentAzimuth,inward,l*M_PI/2);
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

        else if(samplingMethod == "random")
        {
            //Random values generation
            std::default_random_engine re;
            std::uniform_real_distribution<double> unifRadius, unifInclination, unifAzimuth;
            unifRadius = std::uniform_real_distribution<double>(radiusLimits[0],radiusLimits[1]);
            unifInclination = std::uniform_real_distribution<double>(inclinationLimits[0],inclinationLimits[1]);
            unifAzimuth = std::uniform_real_distribution<double>(azimuthLimits[0],azimuthLimits[1]);

            for(int i = 0; i < samplingValue; i++)
            {
                ROS_INFO("Trying to find testing configuration %d",i+1);
                currentRadius = unifRadius(re);
                currentInclination = unifInclination(re);
                currentAzimuth = unifAzimuth(re);

                reachable = false;
                for(int l = 0; l < 4; l++)
                {   
                    currentPose = sphericPose(spherePose,currentRadius,currentInclination,currentAzimuth,inward,l*M_PI/2);
                    if(l >= 1)
                    {
                        ROS_INFO("Trying with yaw offset !");
                    }
                    ROS_INFO("Testing pose %d/%d: ", i, samplingValue);
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
                    i--;
                }
            }
        }

        else
        {
            ROS_ERROR("Invalid sampling parameters !");
            throw std::runtime_error("INVALID PARAMETER");
        }
    }
    
    else if(samplingMethod == "naive" || samplingMethod == "random")
    {
        std::vector<std::vector<double>> jointsLimits;

        if(n.hasParam("jointsLimits"))
        {
            std::vector<double> tmpLowerLimits, tmpUpperlimits;
            n.getParam("jointsLimits/lower",tmpLowerLimits);
            n.getParam("jointsLimits/upper",tmpUpperlimits);
            
            if(tmpLowerLimits.size() != tmpUpperlimits.size() || tmpLowerLimits.size() != robot.getJointsNumber() || tmpUpperlimits.size() != robot.getJointsNumber())
            {
                ROS_WARN("Invalid joints limits, defaulting to robot model joints limits !");
                jointsLimits = robot.getJointsLimits();
            }
            else
            {
                for(int i = 0; i < robot.getJointsNumber(); i++)
                {
                    jointsLimits.push_back({tmpLowerLimits[i],tmpUpperlimits[i]});
                }
            }
        }
        else    //Default to robot model joints limits
        {
            jointsLimits = robot.getJointsLimits(); 
        }

        if(samplingMethod == "naive")
        {
            std::vector<double> Q;
            for(int i = 0; i < samplingValue; i++)
            {
                for(int j = 0; j < robot.getJointsNumber(); j++)
                {
                    if(jointsLimits[j][0] - 2*M_PI*floor(jointsLimits[j][0]/(2*M_PI)) != jointsLimits[j][1] - 2*M_PI*floor(jointsLimits[j][1]/(2*M_PI)))
                    {
                        Q.push_back(jointsLimits[j][0] + (jointsLimits[j][1] - jointsLimits[j][0])*i/(samplingValue - 1));
                    }
                    //Case where the lower and upper joints limits are the same (modulo 2*pi)
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

        else    //samplingMethod == "random"
        {
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
                    //ROS_INFO("%f,%f",jointsLimits[j][0],jointsLimits[j][1]);
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
                    config["sampledConfigurations"].push_back(currentConfiguration);
                }
                else
                {
                    ROS_INFO("Unreachable...");
                    i--;
                }
                currentConfiguration.clear();
            }
        }

    }
    
    else
    {
        ROS_ERROR("Invalid sampling parameters !");
        throw std::runtime_error("INVALID PARAMETER");
    }

    //Close file
    std::ofstream fout(sampledConfigurationsFileName);   
    fout << config;

    ros::shutdown();
    return 0;
}