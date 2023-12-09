#ROS package
import rospkg
rospack = rospkg.RosPack()

urdfFilePath = rospack.get_path("kuka_kr10_support") + "/urdf/kr10r1100sixx.xacro"

flangeLinkName = 'tool0'

### Import routine
from robot_arm_calibration.URDFtoModel import URDFtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,piN,TBaseN,TFlangeN = URDFtoModel(urdfFilePath,flangeLinkName)

from robot_arm_calibration.ParametersTools import saveParameters
import os
saveParameters(os.path.splitext(os.path.abspath(__file__))[0] + "_parameters.yaml",TBase=TBaseN,TFlange=TFlangeN,pi=piN)