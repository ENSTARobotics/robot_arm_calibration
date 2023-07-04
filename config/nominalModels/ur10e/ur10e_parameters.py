urdfFilePath = '/home/demo1/Desktop/robot_arm_tools/config/robots/ur10e_calibration.urdf'

flangeLinkName = 'tool0'

### Import routine
from robot_arm_calibration.URDFtoModel import URDFtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = URDFtoModel(urdfFilePath,flangeLinkName)

