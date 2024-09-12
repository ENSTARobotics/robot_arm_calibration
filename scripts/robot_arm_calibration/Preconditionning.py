#!/usr/bin/python3

import casadi as cs
from liecasadi import SO3

from robot_arm_calibration.AnalyticGeometricModel import (
    mrp_to_matrix,
    quat_to_mrp,
    PARAMETERS_NUMBER
)

### VARIABLES ###
L = 1.0
zRef = 0.904 + 0.4 - 0.006

computedPose = cs.SX.sym('computedPose',6)
measuredPose = cs.SX.sym('measuredPose',6)
aimedPosition = cs.SX.sym('aimedPosition',3)  
jacobian = cs.SX.sym('jacobian',6,PARAMETERS_NUMBER)

# Remark : for more complex preconditionning, we need to use the chain rule :
# P_preconditionned(pi,q) = foP(pi,q)
# JP_preconditionned(pi,q) = J(foP(pi,q)) = JfoP(pi,q)*JP(pi,q) = Jf(P(pi,q))*JP(pi,q)

### SIMPLE PRECONDITIONNING ###

deltaPosition = measuredPose[:3] - computedPose[:3]
deltaPositionFunction = cs.Function('deltaPositionFunction',[computedPose,measuredPose],[deltaPosition])

measuredOrientation = mrp_to_matrix(measuredPose[3:])
computedOrientation = mrp_to_matrix(computedPose[3:])

# Remark : Results obtained using the delta rotation matirx are slightly better than those obtained using the MRP difference directly, certainly due to a faster convergence by doing so.
deltaOrientation = measuredOrientation.T@computedOrientation

deltaOrientationMRP = quat_to_mrp(SO3.from_matrix(deltaOrientation))
deltaOrientationEuler = SO3.from_matrix(deltaOrientation).as_euler()
deltaOrientationLog = SO3.from_matrix(deltaOrientation).log().vec
deltaOrientationLogFunction = cs.Function('deltaOrientationLogFunction',[computedPose,measuredPose],[deltaOrientationLog])

simplePreconditionning = cs.Function('simplePreconditionning',[computedPose,measuredPose],[cs.vertcat(deltaPosition,L*deltaOrientationLog)])
JsimplePreconditionning = cs.jacobian(simplePreconditionning(computedPose,measuredPose),computedPose)

simpleEulerPreconditionning = cs.Function('simpleEulerPreconditionning',[computedPose,measuredPose],[cs.vertcat(deltaPosition,L*deltaOrientationEuler)])
JsimpleEulerPreconditionning = cs.jacobian(simpleEulerPreconditionning(computedPose,measuredPose),computedPose)

eulerPreconditionning = cs.Function('eulerPreconditionning',[computedPose,measuredPose],[cs.vertcat(deltaPosition,deltaOrientationEuler)])

### SPHERIC PRECONDITIONNING ###

deltaMeasuredAimed = measuredPose[:3] - aimedPosition
deltaComputedAimed = computedPose[:3] - aimedPosition

measuredR = cs.norm_2(deltaMeasuredAimed)
computedR = cs.norm_2(deltaComputedAimed)
deltaR = measuredR - computedR

measuredTheta = cs.acos(deltaMeasuredAimed[2] / measuredR)
computedTheta = cs.acos(deltaComputedAimed[2] / computedR)
deltaTheta = measuredTheta - computedTheta

measuredPhi = cs.atan2(deltaMeasuredAimed[1],deltaMeasuredAimed[0])
computedPhi = cs.atan2(deltaComputedAimed[1],deltaComputedAimed[0])
deltaPhi = measuredPhi - computedPhi

sphericPreconditionning = cs.Function('sphericPreconditionning',[computedPose,measuredPose,aimedPosition],[cs.vertcat(deltaR,measuredR*deltaTheta,measuredR*deltaPhi,measuredR*deltaOrientationEuler)])
JsphericalPreconditionning = cs.jacobian(sphericPreconditionning(computedPose,measuredPose,aimedPosition),computedPose)

### JACOBIANS ###

simpleJacobianPreconditionning = cs.Function('simpleJacobianPreconditionning',[jacobian,computedPose,measuredPose],[-JsimplePreconditionning@jacobian])
simpleEulerJacobianPreconditionning = cs.Function('simpleEulerJacobianPreconditionning',[jacobian,computedPose,measuredPose],[-JsimpleEulerPreconditionning@jacobian])
sphericJacobianPreconditionning = cs.Function('sphericJacobianPreconditionning',[jacobian,computedPose,measuredPose,aimedPosition],[-JsphericalPreconditionning@jacobian])