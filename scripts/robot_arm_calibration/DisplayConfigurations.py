#!/usr/bin/python3

#System packages
import sys
import yaml
import glob

#Utility packages
import numpy as np

### Import robot and marker parameters
from robot_arm_calibration.ParametersTools import getIdentifiableParametersPartial

#Custom packages
from robot_arm_calibration.NumericGeometricModel import T, TBaseN, TFlangeN, TToolN, piN, JOINTS_NUMBER, toolName, robotName

### Retreive nominal parameters
parameters0 = getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN)

### Loading calibration poses
print("Loading configurations...")
try:
    fileName = sys.argv[1]
except:
    try:
        fileName = glob.glob("*Configurations.yaml")[0]
    except:
        print("Invalid configurations file name ! Are you in a proper calibrated model folder or have you specified a correct configurations file name ? ")
        sys.exit(-1)

with open(fileName,"r") as file:
    Configurations = yaml.load(file, Loader=yaml.FullLoader)
    Configurations = np.array(Configurations[list(Configurations.keys())[0]]).reshape((-1,JOINTS_NUMBER))[:10]
print("Done !")

print('Number of configurations : ' + str(len(Configurations)))

plotMode = input("Plot mode ? [mesh, axes]")

if(plotMode == "axes"):

    import matplotlib.pyplot as plt

    ### Tool function
    def plotFrame(transformationMatrix, ax = None):
        """! Plot the frame assiated to an homogeneous 4x4 transformation matrix.

        @param transformationMatrix The tranformation matrix.
        @param ax                   Previously defined matplotlib.pyplot axes - default : None.

        @return  The generated or completed matplotlib.pyplot axes.
        """
        if(ax is None):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([1,0,0,0]))[:3],color="r",length=0.1,normalize=True)
        ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([0,1,0,0]))[:3],color="g",length=0.1,normalize=True)
        ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([0,0,1,0]))[:3],color="b",length=0.1,normalize=True)
        ax.scatter(*transformationMatrix[:3,-1],s=5,color="k")
        return(ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Poses = np.empty((0,3))
    for q in Configurations:
        transformationMatrix = T(q,piN)
        Poses = np.vstack((Poses,transformationMatrix[:3,-1]))
        plotFrame(transformationMatrix,ax)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    plt.legend()
    plt.show()

elif(plotMode == "mesh"):

    import subprocess
    import numpy as np
    import trimesh

    import rospkg
    rospack = rospkg.RosPack()

    xacro = "echo $(xacro $(rospack find robot_arm_tools)/config/tools/" + toolName + "/" + robotName + "/" + robotName + "_" + toolName + ".urdf.xacro)"
    xacroProcess = subprocess.run(xacro,shell=True,stdout=subprocess.PIPE)
    XACROOutput = str(xacroProcess.stdout)[2:-3]

    import re
    packages = np.unique(np.array([XACROOutput[int(m.start()):int(m.end())] for m in re.finditer('package://[a-zA-Z_]+/', XACROOutput)]))

    for package in packages:
        XACROOutput = XACROOutput.replace(package,rospack.get_path(package.split("/")[-2]) + "/")

    XACROOutput = XACROOutput.replace("file://","")

    URDFfile = open("/tmp/tmp.urdf", "w")
    URDFfile.write(XACROOutput)
    URDFfile.close()

    from urchin import URDF

    scene = trimesh.Scene()
    cfg = {}

    for configuration in Configurations:
        robot = URDF.load('/tmp/tmp.urdf')

        for i,joint in enumerate(robot.actuated_joints):
            cfg[joint.name] = configuration[i]

        axis = trimesh.creation.axis(origin_size=0.01,axis_radius=0.01,axis_length=0.15,transform = robot.link_fk(cfg=cfg)[robot.links[-1]])
        scene.add_geometry(axis)

        meshes = robot.visual_trimesh_fk(cfg=cfg)

        mesh = trimesh.Trimesh()

        for submesh in meshes:
            pose = meshes[submesh]
            submesh.apply_transform(pose)
            mesh = trimesh.util.concatenate([mesh,submesh])

        newColors = np.array(mesh.visual.face_colors)
        newColors[:,-1] = 150
        mesh.visual = trimesh.visual.create_visual(face_colors=newColors,mesh=mesh)
        
        scene.add_geometry(mesh)

    scene.set_camera((np.pi/3,0.0,3*np.pi/4),3.5,(0,0,0.5))
    scene.show()
    data = scene.save_image()
    from PIL import Image
    rendered = Image.open(trimesh.util.wrap_as_stream(data))
    rendered.save("output.png")

else:
    print("Invalid plot mode !")
    sys.exit(-1)