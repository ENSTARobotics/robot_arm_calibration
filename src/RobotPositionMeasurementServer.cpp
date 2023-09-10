#include "robot_arm_calibration/RobotPositionMeasurementServer.h"

#include <iostream>
#include <fstream>
#include <numeric>

#define MAX_TRIES 5

#define AVG_WINDOW 10
#define MED_WINDOW 10

#define N_THREADS 4

#include <Eigen/QR> 
#include <tf2_eigen/tf2_eigen.h>

void RobotPositionMeasurementServer::pointCallback(const geometry_msgs::PointStamped::ConstPtr& msg, std::string pointLocation, int pointIndex)
{
    //Escape eventual failed measurements
    if(abs(msg->point.x) > 9000 || abs(msg->point.y) > 9000 || abs(msg->point.z) > 9000)
    {
        return;
    }
    else
    {
        if(pointLocation == "robotBase" && m_robotBasePointsCounter[pointIndex] < AVG_WINDOW)
        {
            if(m_robotBasePointsCounter[pointIndex] == 0)
            {
                m_robotBasePoints[pointIndex] = geometry_msgs::Point();
            }

            m_robotBasePoints[pointIndex].x += msg->point.x/AVG_WINDOW;
            m_robotBasePoints[pointIndex].y += msg->point.y/AVG_WINDOW;
            m_robotBasePoints[pointIndex].z += msg->point.z/AVG_WINDOW;

            m_robotBasePointsCounter[pointIndex]++;
        }
        else if(pointLocation == "robotEndEffector" && m_robotEndEffectorPointsCounter[pointIndex] < AVG_WINDOW)
        {
            if(m_robotEndEffectorPointsCounter[pointIndex] == 0)
            {
                m_robotEndEffectorPoints[pointIndex] = geometry_msgs::Point();
            }

            //Apply robot base transform
            Eigen::Vector4d pointVector;
            pointVector << msg->point.x, msg->point.y, msg->point.z, 1;
            pointVector = m_robotBaseTransform*pointVector;

            m_robotEndEffectorPoints[pointIndex].x += pointVector(0)/AVG_WINDOW;
            m_robotEndEffectorPoints[pointIndex].y += pointVector(1)/AVG_WINDOW;
            m_robotEndEffectorPoints[pointIndex].z += pointVector(2)/AVG_WINDOW;

            m_robotEndEffectorPointsCounter[pointIndex]++;
        }
        else{return;}
    }
}

void RobotPositionMeasurementServer::jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
    if(m_robotJointStateCounter < AVG_WINDOW)
    {
        if(m_robotJointStateCounter == 0)
        {
           m_robotJointState.name = msg->name;
           m_robotJointState.position = std::vector<double>(msg->position.size(),0);
           m_robotJointState.velocity = std::vector<double>(msg->velocity.size(),0);
           m_robotJointState.effort = std::vector<double>(msg->effort.size(),0);
        }

        for(int i = 0; i < m_robotJointState.position.size(); i++)
        {
            m_robotJointState.position[i] += msg->position[m_jointsNamesMapping[i]]/AVG_WINDOW;
        }
        for(int i = 0; i < m_robotJointState.velocity.size(); i++)
        {
            m_robotJointState.velocity[i] += msg->velocity[m_jointsNamesMapping[i]]/AVG_WINDOW;
        }
        for(int i = 0; i < m_robotJointState.effort.size(); i++)
        {
            m_robotJointState.effort[i] += msg->effort[m_jointsNamesMapping[i]]/AVG_WINDOW;
        }

        m_robotJointStateCounter++;
    }
    else{return;}
}

void RobotPositionMeasurementServer::wrenchCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg)
{
    //Escape eventual failed measurements
    if(abs(msg->wrench.force.x) > 9000 || abs(msg->wrench.force.y) > 9000 || abs(msg->wrench.force.z) > 9000 || abs(msg->wrench.torque.x) > 9000 || abs(msg->wrench.torque.y) > 9000 || abs(msg->wrench.torque.z) > 9000)
    {
        return;
    }
    else
    {
        if(m_robotWrenchCounter < AVG_WINDOW)
        {
            if(m_robotWrenchCounter == 0)
            {
                m_robotWrench = geometry_msgs::Wrench();
            }

            m_robotWrench.force.x += msg->wrench.force.x/AVG_WINDOW;
            m_robotWrench.force.y += msg->wrench.force.y/AVG_WINDOW;
            m_robotWrench.force.z += msg->wrench.force.z/AVG_WINDOW;

            m_robotWrench.torque.x += msg->wrench.torque.x/AVG_WINDOW;
            m_robotWrench.torque.y += msg->wrench.torque.y/AVG_WINDOW;
            m_robotWrench.torque.z += msg->wrench.torque.z/AVG_WINDOW;

            m_robotWrenchCounter++;
        }
        else{return;}
    }
}

RobotPositionMeasurementServer::RobotPositionMeasurementServer() : MeasurementServer(), m_measurementsSpinner(N_THREADS,&m_measurementsQueue)
{
    //Assing measurements callback to the dedicated queue. This is done to avoid the callback to be executed in the main thread along the MeasurementServer ros::spin()
    m_measurementsNodeHandle.setCallbackQueue(&m_measurementsQueue);

    //Handle robot end effector points measurements
    if(!m_nodeHandle.getParam("robotEndEffectorPointsTopics",m_robotEndEffectorPointsTopics))
    {
        ROS_ERROR("Unable to retrieve robot end effector points topics !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    m_robotEndEffectorPoints.resize(m_robotEndEffectorPointsTopics.size());
    for(int i = 0; i < m_robotEndEffectorPointsTopics.size(); i++)
    {
        m_robotEndEffectorPointsSubscribers.push_back(m_measurementsNodeHandle.subscribe<geometry_msgs::PointStamped>("/" + m_robotEndEffectorPointsTopics[i], 1, boost::bind(&RobotPositionMeasurementServer::pointCallback, this, _1, "robotEndEffector", i)));
    }

    //Handle robot joint state measurements
    if(!m_nodeHandle.getParam("robotJointStateTopic",m_robotJointStateTopic))
    {
        ROS_WARN("Could not retrieve robot joint state topic name, defaulting to joint_states");
        m_robotJointStateTopic = "joint_states";
    }
    if(!m_nodeHandle.getParam("robotJointStateData",m_robotJointStateData))
    {
        ROS_WARN("Could not retrieve robot joint state measured data, defaulting to position");
        m_robotJointStateData = {"position"};
    }

    double startTime = ros::WallTime::now().toSec();
    while(!m_nodeHandle.getParam("jointsNamesMapping",m_jointsNamesMapping) && ros::WallTime::now().toSec() - startTime < 10.0){continue;}
    if(m_jointsNamesMapping.empty())
    {
        ROS_ERROR("Unable to retrieve joints names mapping !");
        throw std::runtime_error("MISSING PARAMETER");
    } 
    m_robotJointStateSubscriber = m_measurementsNodeHandle.subscribe("/" + m_robotJointStateTopic, 1, &RobotPositionMeasurementServer::jointStateCallback, this);

    //Handle robot flange wrench measurements
    if(!m_nodeHandle.getParam("robotWrenchTopic",m_robotWrenchTopic))
    {
        ROS_WARN("Could not retrieve robot flange wrench topic name, no wrench measurement will be performed !");
        m_robotWrenchTopic = "";
    }
    else
    {
        m_robotWrenchSubscriber = m_measurementsNodeHandle.subscribe("/" + m_robotWrenchTopic, 1, &RobotPositionMeasurementServer::wrenchCallback, this);
    }

    //Handle robot base points measurements
    if(!m_nodeHandle.getParam("robotBasePointsTopics",m_robotBasePointsTopics))
    {
        ROS_WARN("Unable to retrieve robot base points number ! Measurements will be performed according to given robot base offset");
        m_robotBasePointsTopics = std::vector<std::string>();
    }
    m_robotBasePoints.resize(m_robotBasePointsTopics.size());
    for(int i = 0; i < m_robotBasePointsTopics.size(); i++)
    {
        m_robotBasePointsSubscribers.push_back(m_measurementsNodeHandle.subscribe<geometry_msgs::PointStamped>("/" + m_robotBasePointsTopics[i], 1, boost::bind(&RobotPositionMeasurementServer::pointCallback, this, _1, "robotBase", i)));
    }

    //Initialise (reset) all measurements counters
    resetMeasurementsCounters();
}

//TODO !! Average value for force and position measurements !!
bool RobotPositionMeasurementServer::measure()
{
    //Compute robot base transform (and any kind of other static measurement)
    if(m_measurementServerCounter == 0)
    {
        computeRobotBaseTransform();
    }

    //Reset all measurements counters
    resetMeasurementsCounters();

    //Get last robot measurements
    m_measurementsSpinner.start();
    while(std::accumulate(m_robotEndEffectorPointsCounter.begin(), m_robotEndEffectorPointsCounter.end(), 0) < m_robotEndEffectorPoints.size()*AVG_WINDOW || m_robotJointStateCounter < AVG_WINDOW || (m_robotWrenchTopic != "" && m_robotWrenchCounter < AVG_WINDOW))
    {
        continue;
    }
    m_measurementsSpinner.stop();

    //Display measurements
    if(m_measurementServerDisplay)
    {
        for(int i = 0; i < m_robotEndEffectorPoints.size(); i++)
        {
            ROS_INFO("End effector point - %d",i+1);
            ROS_INFO("X: %f, Y: %f, Z: %f", m_robotEndEffectorPoints[i].x, m_robotEndEffectorPoints[i].y, m_robotEndEffectorPoints[i].z);
        }
        
        if(std::find(m_robotJointStateData.begin(),m_robotJointStateData.end(), "position") != m_robotJointStateData.end())
        {
            for(int i = 0; i < m_robotJointState.position.size(); i++)
            {
                ROS_INFO("Position J%d: %f", i+1, m_robotJointState.position[i]);
            }
        }
        
        if(std::find(m_robotJointStateData.begin(),m_robotJointStateData.end(), "velocity") != m_robotJointStateData.end())
        {
            for(int i = 0; i < m_robotJointState.velocity.size(); i++)
            {
                ROS_INFO("Velocity J%d: %f", i+1, m_robotJointState.velocity[i]);
            }
        }
        
        if(std::find(m_robotJointStateData.begin(),m_robotJointStateData.end(), "effort") != m_robotJointStateData.end())
        {
            for(int i = 0; i < m_robotJointState.effort.size(); i++)
            {
                ROS_INFO("Effort J%d: %f", i+1, m_robotJointState.effort[i]);
            }
        }

        if(m_robotWrenchTopic != "")
        {
            ROS_INFO("Wrench values");
            ROS_INFO("FX: %f, FY: %f, FZ: %f, TX: %f, TY: %f, TZ: %f", m_robotWrench.force.x, m_robotWrench.force.y, m_robotWrench.force.z, m_robotWrench.torque.x, m_robotWrench.torque.y, m_robotWrench.torque.z);
        }
    }

    //Open the .csv file for calibration data recording
    std::ofstream myfile(m_measurementServerStorageFolder + "Calibration.csv",std::ios::app);

    ROS_INFO("Open CSV - OK");

    //Record the end effector points measurements
    for(int i = 0; i < m_robotEndEffectorPoints.size(); i++)
    {
        myfile << m_robotEndEffectorPoints[i].x;
        myfile << ",";
        myfile << m_robotEndEffectorPoints[i].y;
        myfile << ",";
        myfile << m_robotEndEffectorPoints[i].z;
        myfile << "\n";
    }

    ROS_INFO("Fill robot end effector points - OK");

    //Record the joints positions values
    if(std::find(m_robotJointStateData.begin(),m_robotJointStateData.end(), "position") != m_robotJointStateData.end())
    {
        for(int i = 0; i < m_robotJointState.position.size(); i++)
        {
            myfile << m_robotJointState.position[i];

            if(i != m_robotJointState.position.size() - 1)
            {
                myfile << ",";
            }
        }
        myfile << "\n";
    }

    //Record the joints velocities values
    if(std::find(m_robotJointStateData.begin(),m_robotJointStateData.end(), "velocity") != m_robotJointStateData.end())
    {
        for(int i = 0; i < m_robotJointState.velocity.size(); i++)
        {
            myfile << m_robotJointState.velocity[i];

            if(i < m_robotJointState.velocity.size() - 1)
            {
                myfile << ",";
            }
        }
        myfile << "\n";
    }

    //Record the joints efforts values
    if(std::find(m_robotJointStateData.begin(),m_robotJointStateData.end(), "effort") != m_robotJointStateData.end())
    {
        for(int i = 0; i < m_robotJointState.effort.size(); i++)
        {
            myfile << m_robotJointState.effort[i];

            if(i < m_robotJointState.effort.size() - 1)
            {
                myfile << ",";
            }
        }
        myfile << "\n";
    }

    ROS_INFO("Fill robot joint state - OK");

    if(m_robotWrenchTopic != "")
    {
        //Record the flange wrench measurement
        myfile << m_robotWrench.force.x;
        myfile << ",";
        myfile << m_robotWrench.force.y;
        myfile << ",";
        myfile << m_robotWrench.force.z;
        myfile << ",";
        myfile << m_robotWrench.torque.x;
        myfile << ",";
        myfile << m_robotWrench.torque.y;
        myfile << ",";
        myfile << m_robotWrench.torque.z;
        myfile << "\n";

        ROS_INFO("Fill robot flange wrench - OK");
    }

    myfile << "---\n";

    //Close the file
    myfile.close();

    ROS_INFO("Close CSV - OK");

	return(true);
}

/*!
 *  \brief Computes the projection of a point on a plane given its normal and a point on the plane.
 *  \param point The point to project.
 *  \param planeNormal The normal of the plane.
 *  \param planePoint A point on the plane.
 */
Eigen::Vector3d planeProjection(Eigen::Vector3d point, Eigen::Vector3d planeNormal, Eigen::Vector3d planePoint)
{
    Eigen::Vector3d v = point - planePoint;
    double dist = planeNormal.transpose()*v;
    return(point - dist*planeNormal);
}

//TODO Custom layout for robot base points !
void RobotPositionMeasurementServer::computeRobotBaseTransform()
{
    //Mandatory transformation matrix 
    //Robot base frame (R) to world frame (W)
    Eigen::Matrix4d TRW, TWR = Eigen::Matrix4d::Identity();

    if(m_robotBasePointsTopics.size() == 0)
    {
        //Robot base frame (R) to world frame (W) - Estimated.
        std::vector<double> tmpMatrix;

        if(!m_nodeHandle.getParam("robotBaseOffset",tmpMatrix))
        {
            ROS_WARN("Unable to retrieve robot base offset, defaulting to identity !");
            //throw std::runtime_error("MISSING PARAMETER");
            
        }
        else
        {
            TWR = Eigen::Map<Eigen::Matrix4d>(tmpMatrix.data()).transpose();     //Warning : Eigen -> column wise !
        }
        TRW = TWR.inverse();
    }

    else
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
        //|                            |     (Robot base reference frame)
        //|(4)______________________(3)|          
        //                
            
        //Get last robot base points measurements
        m_measurementsSpinner.start();
        while(std::accumulate(m_robotBasePointsCounter.begin(), m_robotBasePointsCounter.end(), 0) < m_robotBasePoints.size()*AVG_WINDOW*5)
        {
            continue;
        }
        m_measurementsSpinner.stop();

        for(int i = 0; i < m_robotBasePoints.size(); i++)
        {
            m_robotBasePointsSubscribers[i].shutdown(); //Optional
            
            if(m_measurementServerDisplay)
            {
                ROS_INFO("Base point - %d",i+1);
                ROS_INFO("X: %f, Y: %f, Z: %f", m_robotBasePoints[i].x, m_robotBasePoints[i].y, m_robotBasePoints[i].z);
            }
        }

        //Get robot base pose position
        Eigen::Vector3d baseCentroid = Eigen::Vector3d::Zero();
        for(int i = 0; i < m_robotBasePoints.size() - 1; i++)
        {
            baseCentroid(0) += m_robotBasePoints[i].x/(m_robotBasePoints.size() - 1);
            baseCentroid(1) += m_robotBasePoints[i].y/(m_robotBasePoints.size() - 1);
            baseCentroid(2) += m_robotBasePoints[i].z/(m_robotBasePoints.size() - 1);
        }

        //Find the robot base plane normal vector
        Eigen::MatrixXd robotBaseMatrix(m_robotBasePoints.size(),3);
        Eigen::VectorXd onesVector(m_robotBasePoints.size());
        for(int i = 0; i < m_robotBasePoints.size(); i++)
        {
            robotBaseMatrix(i,0) = m_robotBasePoints[i].x;
            robotBaseMatrix(i,1) = m_robotBasePoints[i].y;
            robotBaseMatrix(i,2) = m_robotBasePoints[i].z;
            onesVector(i) = 1.0;
        }

        //Computing Moore-Penrose pseudo inverse
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(robotBaseMatrix ,Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tolerance = std::numeric_limits<double>::epsilon() * std::max(robotBaseMatrix.cols(), robotBaseMatrix.rows()) *svd.singularValues().array().abs()(0);
	    Eigen::MatrixXd pseudoInverse = svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();

        Eigen::Vector3d robotBaseNormalZ = pseudoInverse*onesVector;
        robotBaseNormalZ /= robotBaseNormalZ.norm();

        if(robotBaseNormalZ(2) < 0)
        {
            robotBaseNormalZ *= -1.0;
        }

        //Find the other axis
        Eigen::Vector3d robotBaseNormalX = planeProjection(robotBaseMatrix.row(0),robotBaseNormalZ,baseCentroid) - planeProjection(robotBaseMatrix.row(m_robotBasePoints.size()-2),robotBaseNormalZ,baseCentroid);
        robotBaseNormalX /= robotBaseNormalX.norm();
        Eigen::Vector3d robotBaseNormalY = robotBaseNormalZ.cross(robotBaseNormalX);

        //Retreive robot base transformation
        
        //Interpolated frame (C) to measurement device reference frame (W)
        Eigen::Matrix4d TWC = Eigen::Matrix4d::Identity();
        TWC.topLeftCorner(3,3) << robotBaseNormalX,robotBaseNormalY,robotBaseNormalZ;
        TWC.topRightCorner(3,1) << baseCentroid;

        //Robot base frame (R) to interpolated frame (C)
        Eigen::Matrix4d TCR = Eigen::Matrix4d::Identity();
        std::vector<double> tmpMatrix;

        if(!m_nodeHandle.getParam("robotBaseOffset",tmpMatrix))
        {
            ROS_WARN("Unable to retrieve robot base offset, defaulting to identity !");
            //throw std::runtime_error("MISSING PARAMETER");
        }
        else
        {
            TCR = Eigen::Map<Eigen::Matrix4d>(tmpMatrix.data()).transpose();    //Warning : Eigen -> column wise !
        }

        //Measurement device reference frame (W) to robot base frame (R)
        TWR = TWC*TCR; 
        TRW = TWR.inverse();
    }    

    //Create robot base transform
    m_robotBaseTransform = TRW;

    //Broadcast the measurement device reference frame
    Eigen::Isometry3d measurementDeviceTransform;
    measurementDeviceTransform.matrix() = TWR;
    geometry_msgs::TransformStamped transformStamped = tf2::eigenToTransform(measurementDeviceTransform);

    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "world";
    transformStamped.child_frame_id = "measurement_device_reference";

    m_tfBroadcaster.sendTransform(transformStamped);
}

int main( int argc, char **argv)
{
	//Initialize ROS objects
	ros::init(argc, argv, "robot_position_measurement_server");

	RobotPositionMeasurementServer server;
        
    ros::spin();
    	
	return(0);
}