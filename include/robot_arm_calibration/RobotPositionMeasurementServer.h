/**
 * \file RobotPositionMeasurementServer.h
 * \brief Header file of the RobotPositionMeasurementServer class.
 *
 * Header file of the RobotPositionMeasurementServer class.
 * Defines the attributes and methods used to trigger calibration measurements for a robot arm
 *
 */

#pragma once

#include <ros/callback_queue.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/JointState.h>

#include "robot_arm_tools/MeasurementServer.h"

#include <Eigen/Geometry>

 /*! \class RobotPositionMeasurementServer
  * \brief Defines the attributes and methods used to trigger calibration measurements for a robot arm
  */
class RobotPositionMeasurementServer:MeasurementServer
{
    public:

        /*!
         * \brief Constructor.
         */
        RobotPositionMeasurementServer();

        /*!
         *  \brief Triggers a calibration measurement
         */
        bool measure();
       
    private:

        /*!
         *  \brief Computes the robot base transform according to the robot base marker points position measurement.
         */
        void computeRobotBaseTransform();

        /*!
         *  \brief Proceeds a robot (base or end effector) point measurement.
         *  \param msg The message containing the measurement.
         *  \param pointLocation The location of the robot point (base or end effector).
         *  \param pointIndex The index of the point.
         */
        void pointCallback(const geometry_msgs::PointStamped::ConstPtr& msg, std::string pointLocation, int pointIndex);

        /*!
         *  \brief Proceeds a robot joint state measurement.
         *  \param msg The message containing the measurement.
         */
        void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg);

        /*!
         *  \brief Proceeds a robot flange wrench measurement.
         *  \param msg The message containing the measurement.
         */
        void wrenchCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg);

        /*!
         *  \brief Resets all measurements counters.
         */
        void resetMeasurementsCounters()
        {
            m_robotEndEffectorPointsCounter = std::vector<int>(m_robotEndEffectorPoints.size(),0);
            m_robotJointStateCounter = 0;

            m_robotBasePointsCounter = std::vector<int>(m_robotBasePoints.size(),0);
            m_robotWrenchCounter = 0;        
        }

        //TODO Full pose measurements handling ?

        std::vector<geometry_msgs::Point> m_robotEndEffectorPoints; /*!< The robot end effector points */
        std::vector<ros::Subscriber> m_robotEndEffectorPointsSubscribers; /*!< The robot end effector points subscribers */
        std::vector<int> m_robotEndEffectorPointsCounter;    /*!< The counter for the number of robot end effector points measurements received */
        std::vector<std::string> m_robotEndEffectorPointsTopics; /*!< The name of the topics on which the robot end effector points will be published */

        
        sensor_msgs::JointState m_robotJointState; /*!< The robot joint state */
        std::vector<std::string> m_robotJointStateData; /*!< The robot joint state measured data : position, velocity, effort */
        ros::Subscriber m_robotJointStateSubscriber;   /*!< The robot joint state subscriber */
        int m_robotJointStateCounter;    /*!< The counter for the number of robot joint state measurements received */
        std::string m_robotJointStateTopic;  /*!< The name of the topic on which the robot joint state will be published - default to "joint_states" */
        
        std::vector<int> m_jointsNamesMapping;  /*!< Mapping between joint states and URDF joints names */     


        std::vector<geometry_msgs::Point> m_robotBasePoints; /*!< The robot base points */
        std::vector<ros::Subscriber> m_robotBasePointsSubscribers; /*!< The robot base points subscribers */
        std::vector<int> m_robotBasePointsCounter;    /*!< The counter for the number of robot base points measurements received */
        std::vector<std::string> m_robotBasePointsTopics; /*!< The name of the topics on which the robot base points will be published - default to empty list (e.g. none) */
        
        Eigen::Matrix4d m_robotBaseTransform; /*!< The robot base pose in the measurement device reference frame */
        tf2_ros::StaticTransformBroadcaster m_tfBroadcaster;    /*!< The tf broadcaster for the measurement device reference frame */


        geometry_msgs::Wrench m_robotWrench; /*!< The robot flange wrench */
        ros::Subscriber m_robotWrenchSubscriber;   /*!< The robot flange wrench subscriber */
        int m_robotWrenchCounter;    /*!< The counter for the number of robot flange wrench measurements received */
        std::string m_robotWrenchTopic;  /*!< The name of the topic on which the robot flange wrench will be published - default to "" (e.g. none) */
        
    
        ros::NodeHandle m_measurementsNodeHandle;  /*!< The ROS node handle for robot measurements  */
        ros::CallbackQueue m_measurementsQueue ; /*!< The callback queue for robot measurements */
        ros::AsyncSpinner m_measurementsSpinner;    /*!< The async. spinner for robot measurements */
};