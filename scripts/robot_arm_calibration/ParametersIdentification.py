#!/usr/bin/python3

# System packages
import sys
import os
import yaml
from multiprocessing import Pool

# Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

# Gradient packages
from jax import jacfwd
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jR

# Optimization packages
# from cyipopt import minimize_ipopt
from scipy.optimize import basinhopping, minimize, least_squares
from pygad import GA

### Import robot and marker parameters
from robot_arm_calibration.ParametersTools import (
    getIdentifiableParametersPartial,
    getIdentifiableParametersFull,
    saveParameters,
)

# Custom packages (needing robot and marker parameters inclusions)
from robot_arm_calibration.CalibrationCSVParsing import calibrationCSVParsing

# from robot_arm_calibration.AnalyticGeometricModel import Tfunc, Pfunc, Jfunc, TBaseN, TFlangeN, TToolN, piN, EEItemsNames, N_EEPOINTS, JOINTS_NUMBER
from robot_arm_calibration.NumericGeometricModel import (
    Tfunc,
    Pfunc,
    Jfunc,
    TBaseN,
    TFlangeN,
    TToolN,
    piN,
    EEItemsNames,
    N_EEPOINTS,
    JOINTS_NUMBER,
    JOINTS_AXIS,
    JOINTS_OFFSET,
    LINKS_TRANSLATIONS,
    LINKS_ROTATIONS,
)

# Remark : Results obtained using the delta rotation matirx are slightly better than those obtained using the MRP difference directly, certainly due to a faster convergence by doing so.
def simplePreconditionning(computedPose, measuredPose, **kwargs):
    L = 1.0  # TODO As parameter
    #delta = measuredPose - computedPose
    #return jnp.hstack((delta[:3], L * delta[3:]))

    delta_position = measuredPose[:3] - computedPose[:3]

    measuredOrientation = jR.from_mrp(measuredPose[3:]).as_matrix()
    computedOrientation = jR.from_mrp(computedPose[3:]).as_matrix()
    delta_orientation = jnp.dot(measuredOrientation.T, computedOrientation)
    delta_orientation = jR.from_matrix(delta_orientation).as_mrp()

    return jnp.append(delta_position,delta_orientation)


JsimplePreconditionning = jacfwd(simplePreconditionning)


def simpleEulerPreconditionning(computedPose, measuredPose, **kwargs):
    L = 1.0  # TODO As parameter
    #delta = measuredPose - computedPose
    #return jnp.hstack((delta[:3], L * delta[3:]))

    delta_position = measuredPose[:3] - computedPose[:3]

    measuredOrientation = jR.from_mrp(measuredPose[3:]).as_matrix()
    computedOrientation = jR.from_mrp(computedPose[3:]).as_matrix()
    delta_orientation = jnp.dot(measuredOrientation.T, computedOrientation)
    delta_orientation = jR.from_matrix(delta_orientation).as_euler("xyz")

    return jnp.append(delta_position,L*delta_orientation)


JsimpleEulerPreconditionning = jacfwd(simpleEulerPreconditionning)


# Remark : Results obtained here slightly differ from the ones obtained by computing the aimed target with the measurements directly. This is likely caused by (1) the faulty measurement of the target position relative to the robot base and (2) the parallelism assumption of the robot base and the target plane.
def partialSphericPreconditionning(computedPose, measuredPose, **kwargs):
    zRef = 0.904 + 0.4 - 0.006  # TODO As parameter
    position = measuredPose[:3]
    orientation = jR.from_mrp(measuredPose[3:]).as_matrix()

    # Find intersection between axis(p,z) and z = zRef plane
    # => (p + t*z)[2] = zRef => t = (zRef-p[2])/z[2]
    # intersection = p + (zRef-p[2]/z[2])*z

    aimedPosition = (
        position + (zRef - position[2] / orientation[2, 2]) * orientation[:, 2]
    )

    return sphericPreconditionning(computedPose, measuredPose, additionnalParameters=aimedPosition)


JpartialSphericPreconditionning = jacfwd(partialSphericPreconditionning)


def sphericPreconditionning(computedPose, measuredPose, **kwargs):

    aimedPosition = kwargs["additionnalParameters"]

    # Spheric coordinates difference : r, theta, phi
    measuredPosition = measuredPose[:3] - aimedPosition
    computedPosition = computedPose[:3] - aimedPosition

    measured_r = jnp.linalg.norm(measuredPosition)
    computed_r = jnp.linalg.norm(computedPosition)
    delta_r = measured_r - computed_r
    delta_theta = jnp.arccos(measuredPosition[2] / measured_r) - jnp.arccos(
        computedPosition[2] / computed_r
    )
    delta_phi = jnp.arctan2(measuredPosition[1], measuredPosition[0]) - jnp.arctan2(
        computedPosition[1], computedPosition[0]
    )

    # Axis-wise orientation difference : euler angles, x, y, z
    measuredOrientation = jR.from_mrp(measuredPose[3:]).as_matrix()
    computedOrientation = jR.from_mrp(computedPose[3:]).as_matrix()
    delta_orientation = jnp.dot(measuredOrientation.T, computedOrientation)

    delta_angles = jR.from_matrix(delta_orientation).as_euler("xyz")

    # TODO Weighting !
    delta_angles = delta_angles.at[2].set(0.1 * delta_angles[2])
    #delta_theta*=0.1
    #delta_phi*=0.1

    return jnp.append(jnp.array([delta_r, delta_theta, delta_phi]), delta_angles)


JsphericPreconditionning = jacfwd(sphericPreconditionning)


# Remark : for more complex preconditionning, we need to use the chain rule :
# P_preconditionned(pi,q) = foP(pi,q)
# JP_preconditionned(pi,q) = J(foP(pi,q)) = JfoP(pi,q)*JP(pi,q) = Jf(P(pi,q))*JP(pi,q)
def simpleJacobianPreconditionning(computedPose, measuredPose, jacobian, **kwargs):
    return np.dot(JsimplePreconditionning(computedPose, measuredPose, **kwargs), jacobian)


def partialSphericJacobianPreconditionning(computedPose, measuredPose, jacobian, **kwargs):
    return np.dot(JpartialSphericPreconditionning(computedPose, measuredPose, **kwargs), jacobian)


def sphericJacobianPreconditionning(
    computedPose, measuredPose, jacobian, **kwargs
):
    return np.dot(
        JsphericPreconditionning(computedPose, measuredPose, **kwargs), jacobian
    )

preconditionningType = "spheric"
defaultPreconditionning = sphericPreconditionning
defaultJacobianPreconditionning = sphericJacobianPreconditionning

def computeDelta(
    parameters, Pij, Jointi, preconditionning=None, additionnalParameters=None
):
    """! Computes the deltas between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The deltas vector.
    """

    computedData = np.vstack([Pfunc(parameters, q) for q in Jointi])

    if not preconditionning is None:
        if additionnalParameters is None:
            delta = np.hstack(
                [
                    preconditionning(computed, measured)
                    for measured, computed in zip(Pij, computedData)
                ]
            )
        else:
            delta = np.hstack(
                [
                    preconditionning(computed, measured, additionnalParameters=additionnal)
                    for measured, computed, additionnal in zip(
                        Pij, computedData, additionnalParameters
                    )
                ]
            )
    else:
        delta = (Pij - computedData).flatten()

    return delta


def computeError(
    parameters, Pij, Jointi, preconditionning=None, additionnalParameters=None
):
    """! Computes the error (i.e. euclidian distance) between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The computed error.
    """

    delta = computeDelta(
        parameters, Pij, Jointi, preconditionning, additionnalParameters
    )
    return np.dot(delta.T, delta)


def computeIdentificationJacobian(
    parameters, Pij, Jointi, jacobianPreconditionning=None, additionnalParameters=None
):
    """! Computes the identifiaction jacobian matrix.

    @param parameters   The current robot parameters vector.
    @param Jointi       The measured joints positions vector.

    @return  The identification jacbobian matrix.
    """

    if jacobianPreconditionning is None:
        return np.vstack([Jfunc(parameters, q) for q in Jointi])
    else:
        if additionnalParameters is None:
            return np.vstack(
                [
                    jacobianPreconditionning(
                        Pfunc(parameters, q), p, Jfunc(parameters, q)
                    )
                    for p, q in zip(Pij, Jointi)
                ]
            )
        else:
            return np.vstack(
                [
                    jacobianPreconditionning(
                        Pfunc(parameters, q), p, Jfunc(parameters, q), additionnalParameters=additionnal
                    )
                    for p, q, additionnal in zip(Pij, Jointi, additionnalParameters)
                ]
            )


def computeIdentificationGradient(
    parameters,
    Pij,
    Jointi,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Computes the identifiaction gradient vector.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The identification jacbobian matrix.
    """

    delta = computeDelta(
        parameters, Pij, Jointi, preconditionning, additionnalParameters
    )
    gradient = 2 * np.dot(
        computeIdentificationJacobian(
            parameters, Pij, Jointi, jacobianPreconditionning, additionnalParameters
        ).T,
        delta,
    )

    # gradient /= np.linalg.norm(gradient)
    return gradient


def computeLearningRate(
    parameters,
    Pij,
    Jointi,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Computes the optimal learning rate for the gradient descend according to Wolfe conditions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The optimal learning rate.
    """

    delta = computeDelta(
        parameters, Pij, Jointi, preconditionning, additionnalParameters
    )
    J = computeIdentificationJacobian(
        parameters, Pij, Jointi, jacobianPreconditionning, additionnalParameters
    )
    JtJ = np.dot(J, J.T)
    alpha = np.dot(np.dot(delta.T, JtJ), delta) / (
        2 * np.dot(np.dot(delta.T, np.dot(JtJ, JtJ)), delta)
    )
    return alpha


globalAlgorithms = {"BH": "BFGS based Basin-Hopping"}


def globalIdentifyParameters(
    initialParameters,
    PijLear,
    JointiLear,
    maxIterations=10,
    algorithmName="BH",
    PijTest=None,
    JointiTest=None,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the specified global algorithm.

    @param initialParameters    The current robot parameters vector.
    @param PijLear              The learning measured positions vector.
    @param JointiLear           The learning measured joints positions vector.
    @param maxIterations        The maximum number of iterations- default = 10.
    @param PijTest              The testing measured positions vector.
    @param JointiTest           The testing measured joints positions vector.

    @return  The optimized parameters vector.
    """

    print(globalAlgorithms[algorithmName] + " algorithm")

    parameters = globals()[algorithmName](
        initialParameters,
        PijLear,
        JointiLear,
        maxIterations,
        PijTest,
        JointiTest,
        preconditionning=preconditionning,
        jacobianPreconditionning=jacobianPreconditionning,
        additionnalParameters=additionnalParameters,
    )

    errorLear = computeError(
        parameters, PijLear, JointiLear, preconditionning, additionnalParameters
    )
    errorTest = None
    if len(JointiTest) != 0:
        errorTest = computeError(
            parameters, PijTest, JointiTest, preconditionning, additionnalParameters
        )

    return (parameters, errorLear, errorTest)


def BH(
    initialParameters,
    PijLear,
    JointiLear,
    maxIterations,
    PijTest,
    JointiTest,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the BFGS based Basin-Hopping algorithm.

    @param initialParameters    The current robot parameters vector.
    @param PijLear              The learning measured positions vector.
    @param JointiLear           The learning measured joints positions vector.
    @param maxIterations        The maximum number of iterations.
    @param algorithm            The identification algorithm to be used.
    @param PijTest              The testing measured positions vector.
    @param JointiTest           The testing measured joints positions vector.

    @return  The optimized parameters vector.
    """

    def function(parameters):
        return (
            computeError(
                parameters, PijLear, JointiLear, preconditionning, additionnalParameters
            ),
            computeIdentificationGradient(
                parameters,
                PijLear,
                JointiLear,
                preconditionning,
                jacobianPreconditionning,
                additionnalParameters,
            ),
        )

    def callback(x, f, accept):
        # Intermediate error
        print("Intermediate error : " + str(f))

        if len(JointiTest) != 0:
            print(
                "Intermediate test error :  "
                + str(
                    computeError(
                        x, PijTest, JointiTest, preconditionning, additionnalParameters
                    )
                )
            )

        return accept

    result = basinhopping(
        function,
        x0=initialParameters,
        niter=maxIterations,
        stepsize=10,
        minimizer_kwargs={"method": "BFGS", "jac": True},
        disp=True,
        callback=callback,
    )
    optimizedParameters = result.x

    return optimizedParameters


localAlgorithms = {
    "GN": "Gauss-Newton",
    "TLS": "Total Least-Squares",
    "GD": "Gradient descent",
    "LM": "Levendberg-Marquardt",
    "BFGS": "Broyden-Fletcher-Goldfarb-Shanno",
}
# TODO Add stochastic gradient descent ? Not relevant here...


def localIdentifyParameters(
    initialParameters,
    PijLear,
    JointiLear,
    stopDelta=1e-8,
    stopAbs=1e-12,
    algorithmName="GN",
    PijTest=None,
    JointiTest=None,
    maxIterations=100,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the specified locally iterative algorithm.

    @param initialParameters   The current robot parameters vector.
    @param PijLear      The learning measured positions vector.
    @param JointiLear   The learning measured joints positions vector.
    @param stopDelta    The relative delta stopping criterion.
    @param stopAbs      The absolute stopping criterion.
    @param algorithm    The identification algorithm to be used.
    @param PijTest      The testing measured positions vector.
    @param JointiTest   The testing measured joints positions vector.
    @param maxIterations  The maximum number of iterations - default = 10.

    @return  The identified parameters vector, final learning error and final testing error.
    """

    print(localAlgorithms[algorithmName] + " algorithm")

    # Errors initialisation
    errorLear = 10**10
    errorLearPrev = 10**10
    errorTest = None

    # Parameters initialisation
    parameters = initialParameters

    for i in range(maxIterations):

        print("Iteration " + str(i + 1))

        parameters = globals()[algorithmName](
            parameters,
            PijLear,
            JointiLear,
            preconditionning=preconditionning,
            jacobianPreconditionning=jacobianPreconditionning,
            additionnalParameters=additionnalParameters,
        )

        # Intermediate error
        errorLearPrev = errorLear
        errorLear = computeError(
            parameters, PijLear, JointiLear, preconditionning, additionnalParameters
        )
        print("Intermediate error :  " + str(errorLear))

        if len(JointiTest) != 0:
            errorTest = computeError(
                parameters, PijTest, JointiTest, preconditionning, additionnalParameters
            )
            print("Intermediate test error :  " + str(errorTest))

        # Stopping criterion
        # print(1 - errorLear/errorLearPrev)
        if errorLear < stopAbs or 1 - errorLear / errorLearPrev < stopDelta:
            break

    return (parameters, errorLear, errorTest)


def GN(
    parameters,
    Pij,
    Jointi,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the Gauss-Newton algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The optimized parameters vector.
    """

    optimizedParameters = parameters - np.dot(
        np.linalg.pinv(
            computeIdentificationJacobian(
                parameters, Pij, Jointi, jacobianPreconditionning, additionnalParameters
            )
        ),
        computeDelta(parameters, Pij, Jointi, preconditionning, additionnalParameters),
    )

    return optimizedParameters


def TLS(
    parameters,
    Pij,
    Jointi,
    covarianceMatrix=np.eye(6 * len(TToolN)),
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the Total Least-Squares algorithm.

    @param parameters       The current robot parameters vector.
    @param Pij              The measured positions vector.
    @param Jointi           The measured joints values vector.
    @param covarianceMatrix The covariance matrix associated to the measurements.

    @return  The optimized parameters vector.
    """

    covarianceMatrix = np.kron(np.eye(N_LEARNING, dtype=int), covarianceMatrix)
    J = computeIdentificationJacobian(
        parameters, Pij, Jointi, jacobianPreconditionning, additionnalParameters
    )
    tJW = np.dot(J.T, np.linalg.inv(covarianceMatrix))
    tJWJ = np.dot(tJW, J)

    optimizedParameters = parameters - np.dot(
        np.dot(np.linalg.inv(tJWJ), tJW),
        computeDelta(parameters, Pij, Jointi, preconditionning, additionnalParameters),
    )

    return optimizedParameters


def GD(
    parameters,
    Pij,
    Jointi,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the gradient descent algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The optimized parameters vector.
    """

    alpha = computeLearningRate(
        parameters,
        Pij,
        Jointi,
        preconditionning,
        jacobianPreconditionning,
        additionnalParameters,
    )
    optimizedParameters = parameters - alpha * computeIdentificationGradient(
        parameters,
        Pij,
        Jointi,
        preconditionning,
        jacobianPreconditionning,
        additionnalParameters,
    )

    return optimizedParameters


def LM(
    parameters,
    Pij,
    Jointi,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the Levendberg-Marquardt algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The optimized parameters vector.
    """

    delta = computeDelta(
        parameters, Pij, Jointi, preconditionning, additionnalParameters
    )
    J = computeIdentificationJacobian(
        parameters, Pij, Jointi, jacobianPreconditionning, additionnalParameters
    )
    JtJ = np.dot(J, J.T)
    alpha = np.dot(np.dot(delta.T, JtJ), delta) / (
        2 * np.dot(np.dot(delta.T, np.dot(JtJ, JtJ)), delta)
    )

    tJJ = np.dot(J.T, J)
    optimizedParameters = parameters - np.dot(
        np.linalg.inv(tJJ + alpha * np.diag(np.diag(tJJ))), np.dot(J.T, delta)
    )

    return optimizedParameters


def BFGS(
    parameters,
    Pij,
    Jointi,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the Broyden-Fletcher-Goldfarb-Shanno algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The optimized parameters vector.
    """

    delta = computeDelta(
        parameters, Pij, Jointi, preconditionning, additionnalParameters
    )
    J = computeIdentificationJacobian(
        parameters, Pij, Jointi, jacobianPreconditionning, additionnalParameters
    )
    gradient = 2 * np.dot(J.T, delta)

    try:
        # Compute direction
        global B
        d = np.dot(np.linalg.inv(B), gradient)
    except:
        # Initialisation step
        B = np.eye(len(parameters))

        # Compute direction
        d = np.dot(np.linalg.inv(B), gradient)

    # Compute learning rate
    alpha = np.dot(np.dot(delta.T, J), d) / np.dot(np.dot(np.dot(d.T, J.T), J), d)

    # Update parameters
    deltaParameters = alpha * d
    optimizedParameters = parameters - alpha * d

    # Update B matrix
    delta = computeDelta(
        optimizedParameters, Pij, Jointi, preconditionning, additionnalParameters
    )
    J = computeIdentificationJacobian(
        optimizedParameters,
        Pij,
        Jointi,
        jacobianPreconditionning,
        additionnalParameters,
    )
    newGradient = 2 * np.dot(J.T, delta)

    deltaGradient = np.array([newGradient - gradient]).T
    B += np.dot(deltaGradient, deltaGradient.T) / np.dot(
        deltaGradient.T, deltaParameters
    ) - np.dot(np.dot(np.dot(B, deltaParameters), deltaParameters.T), B) / np.dot(
        np.dot(deltaParameters.T, B), deltaParameters
    )

    return optimizedParameters


# TODO Test !
localAlgorithmsLib = {
    "CG_scipy": "Conjugate gradient",
    "LM_scipy": "Levendberg-Marquardt",
    "BFGS_scipy": "Broyden-Fletcher-Goldfarb-Shanno",
    "IP_ipopt": "Interior point",
}


def localLibIdentifyParameters(
    initialParameters,
    PijLear,
    JointiLear,
    stopDelta=1e-8,
    stopAbs=1e-12,
    algorithmName="GN",
    PijTest=None,
    JointiTest=None,
    maxIterations=100,
    preconditionning=None,
    jacobianPreconditionning=None,
    additionnalParameters=None,
):
    """! Performs the robot model parameters identification using the specified locally iterative algorithm.

    @param initialParameters   The current robot parameters vector.
    @param PijLear      The learning measured positions vector.
    @param JointiLear   The learning measured joints positions vector.
    @param stopDelta    The relative delta stopping criterion.
    @param stopAbs      The absolute stopping criterion.
    @param algorithm    The optimization algorithm to be used.
    @param PijTest      The testing measured positions vector.
    @param JointiTest   The testing measured joints positions vector.
    @param maxIterations  The maximum number of iterations - default = 10.

    @return  The identified parameters vector, final learning error and final testing error.
    """

    print(localAlgorithmsLib[algorithmName] + " algorithm")

    # Errors initialisation
    errorLear = 0
    errorTest = None

    # Parameters initialisation
    parameters = initialParameters
    result = None

    def function_min(parameters):
        return computeError(
            parameters, PijLear, JointiLear, preconditionning, additionnalParameters
        )

    def jacobian_min(parameters):
        return computeIdentificationGradient(
            parameters,
            PijLear,
            JointiLear,
            preconditionning,
            jacobianPreconditionning,
            additionnalParameters,
        )

    def function_res(parameters):
        return computeDelta(
            parameters, PijLear, JointiLear, preconditionning, additionnalParameters
        )

    def jacobian_res(parameters):
        return computeIdentificationJacobian(
            parameters,
            PijLear,
            JointiLear,
            jacobianPreconditionning,
            additionnalParameters,
        )

    def callback(IntermediateResult):
        print(
            "Intermediate error : "
            + str(
                computeError(
                    IntermediateResult,
                    PijLear,
                    JointiLear,
                    preconditionning,
                    additionnalParameters,
                )
            )
        )

        if len(JointiTest) != 0:
            print(
                "Intermediate test error :  "
                + str(
                    computeError(
                        IntermediateResult,
                        PijTest,
                        JointiTest,
                        preconditionning,
                        additionnalParameters,
                    )
                )
            )

    if algorithmName.split("_")[1] == "scipy":
        method = algorithmName.split("_")[0]

        if method.upper() == "LM":
            result = least_squares(
                function_res,
                x0=initialParameters,
                jac=jacobian_res,
                method=method.lower(),
                ftol=stopDelta,
                xtol=stopAbs,
                verbose=2,
            )

        else:
            result = minimize(
                function_min,
                x0=initialParameters,
                method=method.lower(),
                jac=jacobian_min,
                tol=stopDelta,
                callback=callback,
            )

        optimizedParameters = result.x
        errorLear = computeError(
            optimizedParameters,
            PijLear,
            JointiLear,
            preconditionning,
            additionnalParameters,
        )
        if len(JointiTest) != 0:
            errorTest = computeError(
                optimizedParameters,
                PijTest,
                JointiTest,
                preconditionning,
                additionnalParameters,
            )

        return (parameters, errorLear, errorTest)

    elif algorithmName.split("_")[1] == "ipopt":
        # result = minimize_ipopt(function, x0=initialParameters, jac=jacobian, tol=stopDelta)

        # optimizedParameters = result.x
        # errorLear = result.fun
        # if(len(JointiTest)!=0):
        #     errorTest = computeError(optimizedParameters,PijTest,JointiTest,preconditionning,additionnalParameters)

        print("WARNING - NOT IMPLEMENTED YET")
        return (parameters, errorLear, errorTest)

    else:
        return (parameters, errorLear, errorTest)


### Main loop
if __name__ == "__main__":

    # Timer initialisation
    # import time
    # t0 = time.time()

    ### Retrieve measurements data
    print("Retrieving calibration measurement data...")
    itemsNames = EEItemsNames + ["jointsPositions"]

    try:
        Items = calibrationCSVParsing(itemsNames, "Calibration.csv")
    except:
        print(
            "No calibration measurements CSV ! Are you in a proper calibrated model folder ? "
        )
        sys.exit(-1)
    print("Done !")

    ### Set preconditionning
    if len(TToolN) > 1:
        preconditionning = None
        jacobianPreconditionning = None
        additionnalParameters = None
    else:
        preconditionning = defaultPreconditionning
        jacobianPreconditionning = defaultJacobianPreconditionning
        additionnalParameters = None

    try:
        additionnalParameters = Items[sys.argv[1]]
    except:
        print("No additionnal parameters provided...")

    estimateTBaseOffset = 0
    try:
        estimateTBaseOffset = int(sys.argv[2])
    except:
        print("Skipping robot base transform estimation...")

    ### Robot base, robot parameters and end effector points transformations estimations
    if len(TToolN) > 1:
        parameters0 = getIdentifiableParametersPartial(TBaseN, TFlangeN, TToolN, piN)
    else:
        parameters0 = getIdentifiableParametersFull(TBaseN, TFlangeN, TToolN, piN)

    # Estimate the transformation between the sensor reference and the robot base
    if estimateTBaseOffset != 0:
        print("Estimating sensor to base transformation...")
        TBaseSensor = np.eye(4)

        if len(TToolN) > 1:
            EEPoints0 = []
            EEPoints1 = []
            for i, configuration in enumerate(Items["jointsPositions"]):
                TBaseFlange = np.dot(
                    TBaseN, np.dot(Tfunc(parameters0, configuration), TFlangeN)
                )
                EEPoints0.extend(
                    np.vstack(
                        [
                            np.dot(
                                np.dot(TBaseFlange, TTool), np.array([[0, 0, 0, 1]]).T
                            )[:-1].T
                            for TTool in TToolN
                        ]
                    )
                )
                for j, item in enumerate(EEItemsNames):
                    EEPoints1.append(Items[item][i])

            EEPoints0 = np.array(EEPoints0)
            EEPoints1 = np.array(EEPoints1)

            # Does not enforce rotation matrix constraints
            # TSensorBase = np.dot(np.hstack([EEPoints1,np.ones((len(EEPoints1),1))]).T,np.linalg.pinv(np.hstack([np.dot(np.dot(TFlangeBase,TTool),np.array([[0,0,0,1]]).T) for TTool in TToolN])))

            # Enforces rotation matrix constraints !
            EEPoints1Centroid = np.mean(EEPoints1, axis=0)
            EEPoints0Centroid = np.mean(EEPoints0, axis=0)

            U, S, V = np.linalg.svd(
                np.dot(
                    (EEPoints0 - EEPoints0Centroid).T, (EEPoints1 - EEPoints1Centroid)
                )
            )

            TSensorBase = np.eye(4)
            TSensorBase[:3, :3] = np.dot(
                np.dot(
                    V.T,
                    np.diag(
                        np.append(np.ones(len(S) - 1), np.linalg.det(np.dot(V.T, U.T)))
                    ),
                ),
                U.T,
            )
            TSensorBase[:3, -1] = EEPoints1Centroid - np.dot(
                TSensorBase[:3, :3], EEPoints0Centroid
            )
            TBaseSensor = np.linalg.inv(TSensorBase)

        else:
            PBaseSensor = []
            RBaseSensor = []
            for i, configuration in enumerate(Items["jointsPositions"]):
                EEPose0 = np.dot(
                    TBaseN, np.dot(Tfunc(parameters0, configuration), TFlangeN)
                )
                EEPose0 = np.dot(EEPose0, TToolN[0])

                EEPose1 = np.eye(4)
                EEPose1[:3, -1] = Items[EEItemsNames[0]][i][:3]
                EEPose1[:3, :3] = R.from_mrp(Items[EEItemsNames[0]][i][3:]).as_matrix()

                # TBaseSensor = TBaseTool * TToolSensor
                DeltaEEPose = np.dot(EEPose0, np.linalg.inv(EEPose1))
                PBaseSensor.append(DeltaEEPose[:3, -1])
                RBaseSensor.append(DeltaEEPose[:3, :3])

            PBaseSensor = np.array(PBaseSensor)
            RBaseSensor = np.array(RBaseSensor)

            # Compute average transformation over all measurements
            PBaseSensor = np.average(PBaseSensor, axis=0)
            RBaseSensor = np.average(RBaseSensor)
            U, _, V = np.linalg.svd(RBaseSensor)
            RBaseSensor = np.dot(U, V)

            TBaseSensor[:3, -1] = PBaseSensor
            TBaseSensor[:3, :3] = RBaseSensor

        print("Sensor to base transformation estimation : \n" + str(TBaseSensor))

        # Transfering measurements in the estimated robot base frame !
        with open("CalibrationCorrected.csv", "wb+") as file:
            for i, jointsPosition in enumerate(Items["jointsPositions"]):
                if len(TToolN) > 1:
                    for j, item in enumerate(EEItemsNames):
                        Items[item][i] = np.dot(
                            TBaseSensor, np.append(Items[item][i], 1).T
                        )[:3]
                        np.savetxt(
                            file,
                            np.round(Items[item][i].reshape(1, -1), 6),
                            delimiter=",",
                            fmt="%1.6f",
                        )
                else:
                    T = np.eye(4)
                    T[:3, -1] = Items[EEItemsNames[0]][i][:3]
                    T[:3, :3] = R.from_mrp(Items[EEItemsNames[0]][i][3:]).as_matrix()
                    Items[EEItemsNames[0]][i] = np.dot(TBaseSensor, T).flatten()
                    np.savetxt(
                        file,
                        np.round(Items[item][i].reshape(1, -1), 6),
                        delimiter=",",
                        fmt="%1.6f",
                    )
                np.savetxt(
                    file,
                    np.round(jointsPosition.reshape(1, -1), 6),
                    delimiter=",",
                    fmt="%1.6f",
                )
                file.write(str.encode("---\n"))

        # print(np.dot(TSensorBase,np.hstack([EEPoints0,np.ones((len(EEPoints0),1))]).T).T[:,:-1] - EEPoints1)

    ### Create learning and testing subsets
    N_MEASUREMENTS = len(Items["jointsPositions"])
    N_LEARNING = min(N_MEASUREMENTS, 2 * len(parameters0))
    N_TESTS = 0

    print("Number of experiments : " + str(N_LEARNING))
    print("Number of tests : " + str(N_TESTS))

    print("Test on first measurement :")
    if len(TToolN) > 1:
        for i, item in enumerate(EEItemsNames):
            print("Delta end effector point " + item)
            print(
                Items[item][0]
                - Pfunc(parameters0, Items["jointsPositions"][0])[3 * i : 3 * i + 3].T[
                    0
                ]
            )
    else:
        if additionnalParameters is None:
            print(
                defaultPreconditionning(
                    Pfunc(parameters0, Items["jointsPositions"][0]),
                    Items[EEItemsNames[0]][0],
                )
            )
        else:
            print(
                defaultPreconditionning(
                    Pfunc(parameters0, Items["jointsPositions"][0]),
                    Items[EEItemsNames[0]][0],
                    additionnalParameters=additionnalParameters[0],
                )
            )

    print("---")

    randomIndex = np.random.randint(0, N_MEASUREMENTS)
    print("Test on random measurement (" + str(randomIndex) + ") :")
    randomIndex = np.random.randint(0, N_MEASUREMENTS)
    if len(TToolN) > 1:
        for i, item in enumerate(EEItemsNames):
            print("Delta end effector point " + item)
            print(
                Items[item][randomIndex]
                - Pfunc(parameters0, Items["jointsPositions"][randomIndex])[
                    3 * i : 3 * i + 3
                ].T[0]
            )
    else:
        if additionnalParameters is None:
            print(
                defaultPreconditionning(
                    Pfunc(parameters0, Items["jointsPositions"][randomIndex]),
                    Items[EEItemsNames[0]][randomIndex],
                )
            )
        else:
            print(
                defaultPreconditionning(
                    Pfunc(parameters0, Items["jointsPositions"][randomIndex]),
                    Items[EEItemsNames[0]][randomIndex],
                    additionnalParameters=additionnalParameters[randomIndex],
                )
            )

    tmp = input("Press enter to continue.")

    if len(TToolN) > 1:
        PijLear = np.empty((N_LEARNING, N_EEPOINTS, 3))
        PijTest = np.empty((N_TESTS, N_EEPOINTS, 3))
    else:
        PijLear = np.empty((N_LEARNING, 6))
        PijTest = np.empty((N_TESTS, 6))

    JointiLear = np.empty((N_LEARNING, JOINTS_NUMBER))
    JointiTest = np.empty((N_TESTS, JOINTS_NUMBER))

    randomIndices = np.random.permutation(N_LEARNING + N_TESTS)
    randomIndices = np.arange(N_LEARNING + N_TESTS)

    # Randomly shuffle measurements between the learning and testing subsets
    indexLear = randomIndices[:N_LEARNING]
    if len(TToolN) > 1:
        for i, item in enumerate(EEItemsNames):
            PijLear[:, i, :] = Items[item][indexLear]
    else:
        PijLear[:, :] = Items[EEItemsNames[0]][indexLear]
    JointiLear = Items["jointsPositions"][indexLear]

    if N_TESTS != 0:
        indexTest = randomIndices[-N_TESTS:]
        if len(TToolN) > 1:
            for i, item in enumerate(EEItemsNames):
                PijTest[:, i, :] = Items[item][indexTest]
        else:
            PijTest[:, :] = Items[EEItemsNames[0]][indexTest]
        JointiTest = Items["jointsPositions"][indexTest]

    # Initial error
    initialError = computeError(
        parameters0, PijLear, JointiLear, preconditionning, additionnalParameters
    )
    print("Initial error : " + str(initialError))

    if len(JointiTest) != 0:
        errorTest = computeError(
            parameters0, PijTest, JointiTest, preconditionning, additionnalParameters
        )
        print("Initial test error : " + str(errorTest))

    ### Retreive identification parameters
    algorithmName = "GN"
    algorithmType = "local"
    try:
        tmpAlgorithmName = sys.argv[3]
        if tmpAlgorithmName in localAlgorithms.keys():
            algorithmType = "local"
        elif tmpAlgorithmName in localAlgorithmsLib.keys():
            algorithmType = "localLib"
        elif tmpAlgorithmName in globalAlgorithms.keys():
            algorithmType = "global"
        else:
            raise ValueError
        algorithmName = tmpAlgorithmName
    except:
        print(
            "No or invalid identification method given, defaulting to "
            + algorithmType
            + " "
            + algorithmName
        )

    if algorithmType == "local" or algorithmType == "localLib":
        stopDelta = 1e-8
        try:
            stopDelta = float(sys.argv[4])
        except:
            print("No stop delta criterion given, defaulting to " + str(stopDelta))

        stopAbs = 1e-12
        try:
            stopAbs = float(sys.argv[5])
        except:
            print("No stop absolute criterion given, defaulting to " + str(stopAbs))

        initialisations = 5
        try:
            initialisations = int(sys.argv[6])
        except:
            print(
                "No initialisations number given, defaulting to " + str(initialisations)
            )

        initialParamters = parameters0

        # Create small initial disturbances...
        if initialisations != 0:

            print("Compute best " + str(initialisations) + " initialisations deltas")
            initialisationDeltas = np.zeros(
                (max(10, 2 * initialisations), len(parameters0))
            )
            initialisationDeltas[1:] = np.random.uniform(
                -0.1, 0.1, (max(10, 2 * initialisations) - 1, len(parameters0))
            )
            print("Initial initialisation deltas : " + str(initialisationDeltas))

            # ...But optimized !
            geneticAlgorithm = GA(
                num_generations=20,
                num_parents_mating=int(max(10, 2 * initialisations) / 2),
                fitness_func=lambda instance, solution, solution_idx: (-1)
                * computeError(
                    parameters0 + solution,
                    PijLear,
                    JointiLear,
                    preconditionning,
                    additionnalParameters,
                ),
                initial_population=initialisationDeltas,
                init_range_low=-1,
                init_range_high=1,
                keep_elitism=initialisations,
                on_generation=lambda instance: print(
                    "Generation : " + str(instance.generations_completed)
                ),
            )
            geneticAlgorithm.run()
            bestInitialisationDeltas = geneticAlgorithm.last_generation_elitism

            print(
                "Run paramrters identification with optimal initialisations deltas : "
                + str(bestInitialisationDeltas)
            )

            # Run parameters identification (in parallel)
            def tmp(x):
                if algorithmType == "local":
                    return localIdentifyParameters(
                        parameters0 + x,
                        PijLear,
                        JointiLear,
                        stopDelta,
                        stopAbs,
                        algorithmName,
                        PijTest,
                        JointiTest,
                        preconditionning=preconditionning,
                        jacobianPreconditionning=jacobianPreconditionning,
                        additionnalParameters=additionnalParameters,
                    )
                else:
                    return localLibIdentifyParameters(
                        parameters0 + x,
                        PijLear,
                        JointiLear,
                        stopDelta,
                        stopAbs,
                        algorithmName,
                        PijTest,
                        JointiTest,
                        preconditionning=preconditionning,
                        jacobianPreconditionning=jacobianPreconditionning,
                        additionnalParameters=additionnalParameters,
                    )

            with Pool(processes=os.cpu_count() - 1) as pool:
                identificationOutputs = pool.map(tmp, bestInitialisationDeltas)

            finalError = 10**10

            # Get best parameters with lowest identification error
            # Remark : further identification is not necessary !
            bestInitialisationDelta = None
            for i, output in enumerate(identificationOutputs):
                if output[1] < finalError:
                    parameters, finalError, finalTestError = output
                    bestInitialisationDelta = bestInitialisationDeltas[i]

            print("Best initialisation delta : " + str(bestInitialisationDelta))

        else:
            if algorithmType == "local":
                parameters, finalError, finalTestError = localIdentifyParameters(
                    initialParamters,
                    PijLear,
                    JointiLear,
                    stopDelta,
                    stopAbs,
                    algorithmName,
                    PijTest,
                    JointiTest,
                    preconditionning=preconditionning,
                    jacobianPreconditionning=jacobianPreconditionning,
                    additionnalParameters=additionnalParameters,
                )
            else:
                parameters, finalError, finalTestError = localLibIdentifyParameters(
                    initialParamters,
                    PijLear,
                    JointiLear,
                    stopDelta,
                    stopAbs,
                    algorithmName,
                    PijTest,
                    JointiTest,
                    preconditionning=preconditionning,
                    jacobianPreconditionning=jacobianPreconditionning,
                    additionnalParameters=additionnalParameters,
                )

    elif algorithmType == "global":
        maxIterations = 10
        try:
            maxIterations = float(sys.argv[4])
        except:
            print(
                "No maximum iterations number given, defaulting to "
                + str(maxIterations)
            )

        parameters, finalError, finalTestError = globalIdentifyParameters(
            parameters0,
            PijLear,
            JointiLear,
            maxIterations,
            algorithmName,
            PijTest,
            JointiTest,
            preconditionning=preconditionning,
            jacobianPreconditionning=jacobianPreconditionning,
            additionnalParameters=additionnalParameters,
        )

    # Final error
    print("Final error : " + str(finalError))

    if len(JointiTest) != 0:
        print("Final test error : " + str(finalTestError))

    ### FILL ROBOT CALIBRATED MODEL

    print("Saving robot calibrated model...")

    # Robot base
    PBase = np.array([parameters[:3]]).T
    RBase = R.from_mrp(parameters[3:6]).as_matrix()
    TBase = np.vstack((np.hstack((RBase, PBase)), np.array([[0, 0, 0, 1]])))

    # Robot geometric parameters
    pi = parameters[6 : 6 + len(piN)]

    # Robot flange and tool
    if len(TToolN) > 1:
        TFlangeList = np.empty(np.shape(TToolN))
        TTool = np.empty(np.shape(TToolN))
        for i, P in enumerate(np.reshape(parameters[6 + len(piN) :], (-1, 3))):
            TTool[i] = np.dot(TFlangeN, TToolN[i])
            TTool[i][:3, -1] = P
            TFlangeList[i] = np.dot(TTool[i], np.linalg.inv(TToolN[i]))
        TFlange = np.average(TFlangeList, axis=0)

        saveParameters(
            os.getcwd() + "/" + algorithmName + "/CalibratedModel.yaml",
            TBase=TBase,
            TFlange=TFlange,
            pi=pi,
            TTool=TTool,
        )
    else:
        PFlangeTool = np.array([parameters[6 + len(piN) : 9 + len(piN)]]).T
        RFlangeTool = R.from_mrp(parameters[9 + len(piN) :]).as_matrix()
        TFlangeTool = np.vstack(
            (np.hstack((RFlangeTool, PFlangeTool)), np.array([[0, 0, 0, 1]]))
        )

        TFlange = np.dot(TFlangeTool, np.linalg.inv(TToolN))
        saveParameters(
            os.getcwd() + "/" + algorithmName + "/CalibratedModel.yaml",
            TBase=TBase,
            TFlange=TFlange,
            pi=pi,
            TTool=TToolN,
        )

    # Additionnal parameters/information
    calibratedModel = {}
    # calibratedModel["TLastLinkTools"] = TTool.flatten().tolist()
    if estimateTBaseOffset != 0:
        # robotBaseOffset = TSensorBase
        calibratedModel["robotBaseOffset"] = (
            np.linalg.inv(TBaseSensor).flatten().tolist()
        )

    calibratedModel["initialError"] = float(initialError)
    calibratedModel["finalError"] = float(finalError)
    calibratedModel["preconditionningType"] = preconditionningType

    with open(os.getcwd() + "/" + algorithmName + "/CalibratedModel.yaml", "a") as file:
        yaml.dump(calibratedModel, file)

    print("Done !")

    # print("-- Parameters identified in " + str(time.time() - t0) + " seconds --")
