from os import dup
import numpy as np

#dUpward = 0.0349+0.010
#dSideward = 0.0034+0.0125
#TTool0_optiTrack = np.array([[[1,0,0,dUpward/np.sqrt(2) - dSideward/np.sqrt(2)],
#                              [0,1,0,dUpward/np.sqrt(2) + dSideward/np.sqrt(2)],
#                              [0,0,1,-0.0112],
#                              [0,0,0,1]]])

TTool0_optiTrack = np.empty((3,4,4))

TTool0_optiTrack[0] = np.eye(4)
TTool0_optiTrack[0][0,-1] = 0.1

TTool0_optiTrack[1] = np.eye(4)
TTool0_optiTrack[1][1,-1] = 0.1

TTool0_optiTrack[2] = np.eye(4)
TTool0_optiTrack[2][2,-1] = 0.1