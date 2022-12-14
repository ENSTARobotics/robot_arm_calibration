/**
 * \file RobotPositionMeasurementServer.h
 * \brief Header file of the RobotPositionMeasurementServer class.
 *
 * Header file of the RobotPositionMeasurementServer class.
 * Defines the attributes and methods used to trigger calibration measurements for a robot arm
 *
 */

#pragma once

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
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
        bool measure(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);
       
    private:

        /*!
         *  \brief Computes the robot base transform according to the robot base markers position measurement.
         */
        void computeRobotBaseTransform();

        std::string m_markerName;   /*!< Type of marker used for the calibration measurements */
        int m_markerPointsNumber;  /*!< Number of reference points on the robot calibration tool */
        int m_robotBasePointsNumber;    /*!< Number of reference points on the robot base calibration tool */

        std::vector<int> m_jointsNamesMapping;  /*!< Mapping between joint states and URDF joints names */

        Eigen::Isometry3d m_robotBaseTransform; /*!< Robot base pose in the OptiTrack reference frame */
};
