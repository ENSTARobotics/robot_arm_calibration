#include "robot_arm_calibration/RobotPositionMeasurementServer.h"

#include <sensor_msgs/JointState.h>

#include <iostream>
#include <fstream>
#include <numeric>

#define MAX_TRIES 5
#define AVG_WINDOW 5

#include <Eigen/QR> 

RobotPositionMeasurementServer::RobotPositionMeasurementServer() : MeasurementServer()
{
    //Retrieve robot base calibration tool reference points number
    std::string tmp;
    if(!m_nodeHandle.getParam("mocap_node/rigid_bodies/0/marker_set_size",tmp))
    {
        ROS_ERROR("Unable to retrieve robot base points number !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    m_robotBasePointsNumber = stoi(tmp);

    //Retrieve calibration tool name and reference points number
    if(!m_nodeHandle.getParam("toolName",m_markerName))
    {
        ROS_ERROR("Unable to retrieve marker tool name !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    
    if(m_markerName == "marker2D")
    {
        if(!m_nodeHandle.getParam("mocap_node/rigid_bodies/1/marker_set_size",tmp))
        {
            ROS_ERROR("Unable to retrieve reference points number !");
            throw std::runtime_error("MISSING PARAMETER");
        }
        m_markerPointsNumber = stoi(tmp);
    }

    else if(m_markerName == "marker3D")
    {
        if(!m_nodeHandle.getParam("mocap_node/rigid_bodies/2/marker_set_size",tmp))
        {
            ROS_ERROR("Unable to retrieve reference points number !");
            throw std::runtime_error("MISSING PARAMETER");
        }
        m_markerPointsNumber = stoi(tmp);
    }

    else
    {
        ROS_ERROR("Invalid tool for calibration measurements !");
        throw std::runtime_error("INVALID TOOL");
    }

    double startTime = ros::WallTime::now().toSec();
    while(!m_nodeHandle.getParam("jointsNamesMapping",m_jointsNamesMapping) && ros::WallTime::now().toSec() - startTime < 10.0){}
    if(m_jointsNamesMapping.empty())
    {
        ROS_ERROR("Unable to retrieve joints names mapping !");
        throw std::runtime_error("MISSING PARAMETER");
    }  

    computeRobotBaseTransform(); 
}

bool RobotPositionMeasurementServer::measure(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
    //Get last robot base reference points measurements
    std::vector<geometry_msgs::Point> robotBase, referencePoints;
    geometry_msgs::Point point;

    bool measurementOK = false;
    int counterTries = 0;

    while(!measurementOK && counterTries < MAX_TRIES)
    {
        measurementOK = true;

        /*//Robot Base
        for(int i = 0; i < m_robotBasePointsNumber; i++)
        {
            point = ros::topic::waitForMessage<geometry_msgs::PointStamped>("/mocap_node/robotBase"+std::to_string(i+1))->point;

            if(abs(point.x) > 9000 || abs(point.y) > 9000 || abs(point.z) > 9000)
            {
                counterTries++;
                measurementOK = false;
                break;
            }

            if(m_measurementServerDisplay)
            {
                ROS_INFO("Robot Base - %d",i+1);
                ROS_INFO("X: %f, Y: %f, Z: %f", point.x, point.y, point.z);
            }
            
            robotBase.push_back(point);
        }
        if(!measurementOK){continue;}*/

        //Reference points
        for(int i = 0; i < m_markerPointsNumber; i++)
        {
            point = ros::topic::waitForMessage<geometry_msgs::PointStamped>("/mocap_node/" + m_markerName + std::to_string(i+1))->point;

            if(abs(point.x) > 9000 || abs(point.y) > 9000 || abs(point.z) > 9000)
            {
                counterTries++;
                measurementOK = false;
                break;
            }

            if(m_measurementServerDisplay)
            {
                ROS_INFO("Reference Point - %d",i+1);
                ROS_INFO("X: %f, Y: %f, Z: %f", point.x, point.y, point.z);
            }
            
            //Apply robot base transform
            Eigen::Vector4d pointVector;
            pointVector << point.x, point.y, point.z, 1;
            pointVector = m_robotBaseTransform*pointVector;

            point.x = pointVector(0);
            point.y = pointVector(1);
            point.z = pointVector(2);

            referencePoints.push_back(point);
        }
        if(!measurementOK){continue;}
    }
    if(!measurementOK && counterTries == MAX_TRIES)
    {
        ROS_WARN("Skipping un-measurable pose !");
        return(true);
    }

    //Get last joints angles values measurement
    boost::shared_ptr<const sensor_msgs::JointState> jointState = ros::topic::waitForMessage<sensor_msgs::JointState>("/joint_states");
    ROS_INFO("Joints values");

    if(m_measurementServerDisplay)
    {
        for(int i = 0; i < jointState->position.size(); i++)
        {
            ROS_INFO("J%d: %f", i+1, jointState->position[m_jointsNamesMapping[i]]);
        }
    }

    //Open the .csv file for calibration data recording
    std::ofstream myfile(m_measurementServerStorageFolder + "Calibration.csv",std::ios::app);

    ROS_INFO("Open CSV - OK");

    /*//Record the robot base measurements
    for(int i = 0; i < m_robotBasePointsNumber; i++)
    {
        myfile << robotBase[i].x;
        myfile << ",";
        myfile << robotBase[i].y;
        myfile << ",";
        myfile << robotBase[i].z;
        myfile << "\n";
    }

    ROS_INFO("Fill Robot Base - OK");*/

    //Record the reference points measurements
    for(int i = 0; i < m_markerPointsNumber; i++)
    {
        myfile << referencePoints[i].x;
        myfile << ",";
        myfile << referencePoints[i].y;
        myfile << ",";
        myfile << referencePoints[i].z;
        myfile << "\n";
    }

    ROS_INFO("Fill Reference Points - OK");

    //Record the joints angles values
    for(int i = 0; i < jointState->position.size(); i++)
    {
        myfile << jointState->position[m_jointsNamesMapping[i]];

        if(i != jointState->position.size() - 1)
        {
            myfile << ",";
        }
    }

    ROS_INFO("Fill joints values - OK");

    myfile << "\n---\n";

    //Close the file
    myfile.close();

    ROS_INFO("Close CSV - OK");

	return(true);
}

Eigen::Vector3d planeProjection(Eigen::Vector3d point, Eigen::Vector3d planeNormal, Eigen::Vector3d planePoint)
{
    Eigen::Vector3d v = point - planePoint;
    double dist = planeNormal.transpose()*v;
    return(point - dist*planeNormal);
}

void RobotPositionMeasurementServer::computeRobotBaseTransform()
{
    // ____________________________
    //|(6)                      (1)|
    //|                            |  
    //|                      (7)   |
    //|                            |
    //|             _|_       (2)  |  
    //|  (5)         |             |      Y
    //|                            |      |
    //|                            |     -|-- X
    //|                            |
    //|(4)______________________(3)|          
    //                
        
    //Get last robot base reference points measurements
    std::vector<geometry_msgs::Point> robotBaseMarkers(m_robotBasePointsNumber), tmpRobotBaseMarkers(m_robotBasePointsNumber);
    geometry_msgs::Point point;

    bool measurementOK = false;
    int counterTries = 0, counterMeasurements = 0;

    for(int i = 0; i < AVG_WINDOW; i++)
    {        
        while(!measurementOK && counterTries < MAX_TRIES)
        {
            measurementOK = true;

            //Robot Base
            for(int j = 0; j < m_robotBasePointsNumber; j++)
            {
                point = ros::topic::waitForMessage<geometry_msgs::PointStamped>("/mocap_node/robotBase"+std::to_string(j+1))->point;

                if(abs(point.x) > 9000 || abs(point.y) > 9000 || abs(point.z) > 9000)
                {
                    counterTries++;
                    measurementOK = false;
                    break;
                }
                
                tmpRobotBaseMarkers[j].x = point.x;
                tmpRobotBaseMarkers[j].y = point.y;
                tmpRobotBaseMarkers[j].z = point.z;
            }
        }

        if(counterTries < MAX_TRIES)
        {
            counterMeasurements++;
            for(int j = 0; j < m_robotBasePointsNumber; j++)
            {
                robotBaseMarkers[j].x += tmpRobotBaseMarkers[j].x;
                robotBaseMarkers[j].y += tmpRobotBaseMarkers[j].y;
                robotBaseMarkers[j].z += tmpRobotBaseMarkers[j].z;
            }  
        }

        counterTries = 0;
        measurementOK = false;
    }

    for(int j = 0; j < m_robotBasePointsNumber; j++)
    {
        robotBaseMarkers[j].x /= counterMeasurements;
        robotBaseMarkers[j].y /= counterMeasurements;
        robotBaseMarkers[j].z /= counterMeasurements;
    }

    //Get robot base pose position
    Eigen::Vector3d baseCentroid;
    for(int i = 0; i < robotBaseMarkers.size() - 1; i++)
    {
        baseCentroid(0) += robotBaseMarkers[i].x/(robotBaseMarkers.size() - 1);
        baseCentroid(1) += robotBaseMarkers[i].y/(robotBaseMarkers.size() - 1);
        baseCentroid(2) += robotBaseMarkers[i].z/(robotBaseMarkers.size() - 1);
    }

    //Find the robot base plane normal vector
    Eigen::MatrixXd robotBaseMatrix(robotBaseMarkers.size(),3);
    Eigen::VectorXd onesVector(robotBaseMarkers.size());
    for(int i = 0; i < robotBaseMarkers.size(); i++)
    {
        robotBaseMatrix(i,0) = robotBaseMarkers[i].x;
        robotBaseMatrix(i,1) = robotBaseMarkers[i].y;
        robotBaseMatrix(i,2) = robotBaseMarkers[i].z;
        onesVector(i) = 1.0;
    }

    Eigen::Vector3d robotBaseNormalZ = robotBaseMatrix.completeOrthogonalDecomposition().pseudoInverse()*onesVector;
    robotBaseNormalZ /= robotBaseNormalZ.norm();

    if(robotBaseNormalZ(2) < 0)
    {
        robotBaseNormalZ *= -1.0;
    }

    //Find the other axis
    //TODO Custom layout ?
    Eigen::Vector3d robotBaseNormalX = planeProjection(robotBaseMatrix.row(0),robotBaseNormalZ,baseCentroid) - planeProjection(robotBaseMatrix.row(robotBaseMarkers.size()-2),robotBaseNormalZ,baseCentroid);
    robotBaseNormalX /= robotBaseNormalX.norm();
    Eigen::Vector3d robotBaseNormalY = robotBaseNormalZ.cross(robotBaseNormalX);

    //Retreive robot base transformation
    
    //Interpolated frame (C) to OptiTrack reference frame (W)
    Eigen::Matrix4d TWC = Eigen::MatrixXd::Identity(4,4);
    TWC.topLeftCorner(3,3) << robotBaseNormalX,robotBaseNormalY,robotBaseNormalZ;
    TWC.topRightCorner(3,1) << baseCentroid;

    //Robot base frame (R) to interpolated frame (C)
    std::vector<double> tmpMatrix;

    if(!m_nodeHandle.getParam("robotBaseOffset",tmpMatrix))
    {
        ROS_ERROR("Unable to retrieve robot base offset !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    Eigen::Matrix4d TCR = Eigen::Map<Eigen::Matrix4d>(tmpMatrix.data());

    //Robot base frame(R) to Optitrack reference frame (W)
    Eigen::Matrix4d robotBaseTransform = (TWC*TCR).inverse();

    //Create robot base transform
    m_robotBaseTransform.linear() = robotBaseTransform.topLeftCorner(3,3);
    m_robotBaseTransform.translation() = robotBaseTransform.topRightCorner(3,1);
}

int main( int argc, char **argv)
{
	//Initialize ROS objects
	ros::init(argc, argv, "robot_position_measurement_server");

	RobotPositionMeasurementServer server;
        
    ros::spin();
    	
	return(0);
}
