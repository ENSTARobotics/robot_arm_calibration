#!/usr/bin/python3

import numpy as np

toolDepth = 0.0112

#dX,dY
d = np.array([[0.0265,-0.0615],
              [-0.0412,-0.0629],
              [0.0449,-0.0159],
              [0.0602,0.0159],
              [0.0494,0.0562],
              [-0.0412,0.0612]])

TTool0_body = np.array([[1/np.sqrt(2),1/np.sqrt(2),0,0],
                   [-1/np.sqrt(2),1/np.sqrt(2),0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
TTool0_marker = np.array([np.dot(TTool0_body,np.array([[1,0,0,item[0]],
                                    [0,1,0,item[1]],
                                    [0,0,1,toolDepth],
                                    [0,0,0,1]])) for item in d])

N_REFERENCEPOINTS = len(d)
markerItemsNames = ["optiTrackPose" + str(i) for i in range(len(d))]